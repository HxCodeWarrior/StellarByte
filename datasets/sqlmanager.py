import threading
import sqlite3
import json
import os
from tqdm import tqdm
from typing import Generator, List, Optional, Dict, Any
import logging

class SQLiteConnectionPool:
    """SQLite连接池，避免频繁创建和关闭连接"""
    _connections = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_connection(cls, db_path):
        with cls._lock:
            thread_id = threading.get_ident()
            if db_path not in cls._connections:
                cls._connections[db_path] = {}
            
            if thread_id not in cls._connections[db_path]:
                cls._connections[db_path][thread_id] = sqlite3.connect(
                    db_path, check_same_thread=False
                )
            
            return cls._connections[db_path][thread_id]
    
    @classmethod
    def close_all(cls):
        with cls._lock:
            for db_path, connections in cls._connections.items():
                for conn in connections.values():
                    try:
                        conn.close()
                    except:
                        pass
            cls._connections = {}
            
class SQLiteDatabaseManager:
    """
    SQLite数据库管理类，负责数据库的创建、数据插入和流式读取
    """
    
    def __init__(self, db_path: str):
        """
        初始化数据库管理器
        
        参数:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.thread_local = threading.local()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def connect(self) -> None:
        """连接到数据库，使用连接池"""
        if not hasattr(self.thread_local, 'connection') or self.thread_local.connection is None:
            try:
                self.thread_local.connection = SQLiteConnectionPool.get_connection(self.db_path)
                self.logger.info(f"线程 {threading.get_ident()} 成功连接到数据库: {self.db_path}")
            except sqlite3.Error as e:
                self.logger.error(f"线程 {threading.get_ident()} 连接数据库失败: {e}")
                raise
        return self.thread_local.connection
    
    def disconnect(self) -> None:
        """断开数据库连接"""
        if hasattr(self.thread_local, 'connection') and self.thread_local.connection:
            self.thread_local.connection.close()
            self.thread_local.connection = None
            self.logger.info(f"线程 {threading.get_ident()} 数据库连接已关闭")
    
    def create_table(self, table_name: str = "documents", columns: str = "text") -> None:
        """
        创建数据表
        
        参数:
            table_name: 表名
            columns: 字段名
        """
        db_connected = self.connect()
            
        try:
            cursor = db_connected.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {columns} TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            db_connected.commit()
            self.logger.info(f"表 '{table_name}' 创建成功")
        except sqlite3.Error as e:
            self.logger.error(f"创建表失败: {e}")
            raise
    
    def insert_data(
        self,
        data_file_path: str,
        table_name: str = "documents",
        columns: str = "text",
        batch_size: int = 1000
    ) -> int:
        """
        从JSONL文件导入数据到数据库，仅使用指定的文本字段

        参数:
            jsonl_file_path: JSONL文件路径
            table_name: 表名
            columns: 数据库中的文本字段名
            batch_size: 批量插入大小

        返回:
            插入的行数
        """
        db_connected = self.connect()

        inserted_count = 0
        batch_data = []

        def process_batch():
            """处理当前批次的数据"""
            nonlocal inserted_count
            if not batch_data:
                return

            try:
                cursor = db_connected.cursor()
                placeholders = ", ".join(["?"] * len(batch_data[0]))
                cursor.executemany(
                    f"INSERT INTO {table_name} ({columns}, metadata) VALUES ({placeholders})",
                    batch_data
                )
                db_connected.commit()
                inserted_count += len(batch_data)
                self.logger.info(f"已插入 {inserted_count} 条记录")
                batch_data.clear()
            except sqlite3.Error as e:
                self.logger.error(f"批量插入失败: {e}")
                raise

        try:
            with open(data_file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)

                        # 获取文本字段
                        if columns not in data:
                            self.logger.warning(f"第 {line_num} 行缺少字段 '{columns}'，跳过")
                            continue

                        text = str(data[columns]).strip()
                        if not text:
                            self.logger.warning(f"第 {line_num} 行文本为空，跳过")
                            continue

                        # 准备元数据（排除文本字段本身）
                        metadata = {k: v for k, v in data.items() if k != columns}

                        # 添加到批次
                        batch_data.append((text, json.dumps(metadata)))

                        # 达到批次大小时插入
                        if len(batch_data) >= batch_size:
                            process_batch()

                    except json.JSONDecodeError:
                        self.logger.warning(f"第 {line_num} 行JSON解析错误，跳过")
                        continue
                    except Exception as e:
                        self.logger.warning(f"第 {line_num} 行处理错误: {e}，跳过")
                        continue

            # 处理最后一批数据
            process_batch()

            self.logger.info(f"总共插入 {inserted_count} 条记录")
            return inserted_count

        except Exception as e:
            self.logger.error(f"导入JSONL数据失败: {e}")
            raise

    def stream_text_data(
        self, 
        table_name: str = "documents", 
        columns: str = "text",
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> Generator[str, None, None]:
        """
        流式读取数据库中的文本数据
        
        参数:
            table_name: 表名
            columns: 文本字段名
            batch_size: 每次读取的批量大小
            
        返回:
            文本数据生成器
        """
        db_connected = self.connect()
            
        cursor = db_connected.cursor()
        
        # 获取总行数
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        
        self.logger.info(f"开始流式读取 {total_rows} 条记录")
        
        # 创建进度条（如果需要）
        pbar = tqdm(total=total_rows, desc="读取数据", unit="行", disable=not show_progress)
        
        # 使用基于ID的范围查询而不是OFFSET，提高性能
        min_id = 0
        max_id = 0
        
        # 获取最小和最大ID
        cursor.execute(f"SELECT MIN(id), MAX(id) FROM {table_name}")
        min_max = cursor.fetchone()
        min_id = min_max[0] if min_max[0] is not None else 0
        max_id = min_max[1] if min_max[1] is not None else 0
        
        current_id = min_id
        
        # 分批读取数据
        while current_id <= max_id:
            cursor.execute(
                f"SELECT id, {columns} FROM {table_name} WHERE id >= ? ORDER BY id LIMIT {batch_size}", 
                (current_id,)
            )
            rows = cursor.fetchall()
            
            if not rows:
                break
                
            for row in rows:
                current_id = row[0] + 1  # 准备下一批的起始ID
                if row[1]:  # 确保文本不为空
                    pbar.update(1)
                    yield row[1]
        
        pbar.close()
        self.logger.info("流式读取完成")
    
    def get_table_info(self, table_name: str = "documents") -> Dict[str, Any]:
        """
        获取表信息
        
        参数:
            table_name: 表名
            
        返回:
            表信息字典
        """
        db_connected = self.connect()
            
        cursor = db_connected.cursor()
        
        # 获取表结构
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # 获取行数
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        # 获取示例数据
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_data = cursor.fetchall()
        
        return {
            "table_name": table_name,
            "columns": columns,
            "row_count": row_count,
            "sample_data": sample_data
        }
    
    def optimize_database(self) -> None:
        """优化数据库性能"""
        db_connected = self.connect()
            
        try:
            cursor = db_connected.cursor()
            
            # 启用WAL模式以提高并发性能
            cursor.execute("PRAGMA journal_mode=WAL")
            
            # 设置同步模式为NORMAL以提高写入性能
            cursor.execute("PRAGMA synchronous=NORMAL")
            
            # 设置缓存大小
            cursor.execute("PRAGMA cache_size=-2000")  # 2MB缓存
            
            db_connected.commit()
            self.logger.info("数据库优化完成")
        except sqlite3.Error as e:
            self.logger.error(f"数据库优化失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()

if __name__ == "__main__":
    # 测试代码
    db_path = './datasets/test/test.db'
    data_path = "./datasets/test/test_train.jsonl"
    db_manager = SQLiteDatabaseManager(db_path)
    db_manager.connect()
    db_manager.create_table('test_data', 'text')
    db_manager.insert_data(
        data_file_path=data_path,
        table_name='test_data',
        columns='text'
    )
    db_manager.optimize_database()

    # 获取表信息
    table_info = db_manager.get_table_info("test_data")
    print(f"表名: {table_info['table_name']}")
    print(f"行数: {table_info['row_count']}")
    print("列信息:")
    for col in table_info['columns']:
        print(f"  {col[1]} ({col[2]})")
    
    # 流式读取数据示例
    count = 0
    for text in db_manager.stream_text_data("test_data", "text", 500):
        count += 1
        if count <= 3:  # 只打印前3条记录
            print(f"记录 {count}: {text[:100]}...")
    
    print(f"总共读取了 {count} 条记录")

    db_manager.disconnect()
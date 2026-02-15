import asyncio
import logging
import os
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any

import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class PostgresConfig:
    """PostgreSQL configuration settings"""

    host: str = "localhost"
    port: int = 5432
    database: str = "liquide_db"
    user: str = "postgres"
    password: str = "postgres"
    min_connections: int = 1
    max_connections: int = 10
    ssl_mode: str = "prefer"
    connect_timeout: int = 10
    command_timeout: int = 30

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        """Create configuration from environment variables"""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "liquide_db"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            min_connections=int(os.getenv("POSTGRES_MIN_CONNECTIONS", "1")),
            max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "10")),
            ssl_mode=os.getenv("POSTGRES_SSL_MODE", "prefer"),
            connect_timeout=int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10")),
            command_timeout=int(os.getenv("POSTGRES_COMMAND_TIMEOUT", "30")),
        )

    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_async_connection_string(self) -> str:
        """Get async PostgreSQL connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_connection_params(self) -> dict[str, Any]:
        """Get connection parameters as dictionary"""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "sslmode": self.ssl_mode,
            "connect_timeout": self.connect_timeout,
            "options": f"-c statement_timeout={self.command_timeout}s",
        }


class PostgresConnectionManager:
    """Manages PostgreSQL connections for both sync and async operations"""

    def __init__(self, config: PostgresConfig):
        self.config = config
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()

    async def get_pool(self) -> asyncpg.Pool:
        """Get or create async connection pool"""
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        self.config.get_async_connection_string(),
                        min_size=self.config.min_connections,
                        max_size=self.config.max_connections,
                        command_timeout=self.config.command_timeout,
                        max_inactive_connection_lifetime=60.0,
                        timeout=self.config.connect_timeout,
                        statement_cache_size=0,
                    )
                    logger.info(f"Created PostgreSQL connection pool with {self.config.max_connections} connections")
        return self._pool

    async def close_pool(self):
        """Close the connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Closed PostgreSQL connection pool")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        pool = await self.get_pool()
        conn = await pool.acquire()
        try:
            yield conn
        finally:
            await pool.release(conn)

    @asynccontextmanager
    async def new_connection(self):
        """Open and close a dedicated connection for a single operation."""
        conn = await asyncpg.connect(
            self.config.get_async_connection_string(),
            timeout=self.config.connect_timeout,
        )
        try:
            yield conn
        finally:
            await conn.close()

    @contextmanager
    def get_sync_connection(self):
        """Get a synchronous connection"""
        conn = psycopg2.connect(**self.config.get_connection_params())
        try:
            yield conn
        finally:
            conn.close()

    async def execute(self, query: str, *args, **kwargs):
        """Execute a query asynchronously"""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args, timeout=self.config.command_timeout, **kwargs)

    async def fetch(self, query: str, *args, **kwargs):
        """Fetch results asynchronously"""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args, timeout=self.config.command_timeout, **kwargs)

    async def fetchrow(self, query: str, *args, **kwargs):
        """Fetch a single row asynchronously"""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args, timeout=self.config.command_timeout, **kwargs)

    async def fetchval(self, query: str, *args, **kwargs):
        """Fetch a single value asynchronously"""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args, timeout=self.config.command_timeout, **kwargs)

    def execute_sync(self, query: str, *args, **kwargs):
        """Execute a query synchronously"""
        with self.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, *args, **kwargs)
                conn.commit()
                return cur.fetchall()

    def fetch_sync(self, query: str, *args, **kwargs):
        """Fetch results synchronously"""
        with self.get_sync_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, *args, **kwargs)
                return cur.fetchall()


# Default configuration instance
default_config = PostgresConfig()

# Default connection manager instance
default_manager = PostgresConnectionManager(default_config)


async def test_connection(config: PostgresConfig | None = None) -> bool:
    """Test PostgreSQL connection"""
    if config is None:
        config = default_config

    try:
        manager = PostgresConnectionManager(config)
        async with manager.get_connection() as conn:
            result = await conn.fetchval("SELECT 1")
            logger.info("PostgreSQL connection test successful")
            return result == 1
    except Exception as e:
        logger.error(f"PostgreSQL connection test failed: {e}")
        return False


def create_tables_sync(config: PostgresConfig | None = None):
    """Create basic tables for the Liquide agent"""
    if config is None:
        config = default_config

    tables_sql = """
    CREATE TABLE IF NOT EXISTS conversations (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255) UNIQUE NOT NULL,
        customer_name VARCHAR(255),
        customer_phone VARCHAR(20),
        customer_email VARCHAR(255),
        store_id VARCHAR(100),
        store_name VARCHAR(255),
        order_type VARCHAR(50),
        order_status VARCHAR(50) DEFAULT 'pending',
        total_amount DECIMAL(10,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS conversation_messages (
        id SERIAL PRIMARY KEY,
        conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
        message_type VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB
    );

    CREATE TABLE IF NOT EXISTS orders (
        id SERIAL PRIMARY KEY,
        conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
        order_number VARCHAR(100) UNIQUE,
        items JSONB NOT NULL,
        delivery_address TEXT,
        delivery_instructions TEXT,
        scheduled_time TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
    CREATE INDEX IF NOT EXISTS idx_conversations_store_id ON conversations(store_id);
    CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON conversation_messages(conversation_id);
    CREATE INDEX IF NOT EXISTS idx_orders_conversation_id ON orders(conversation_id);
    """

    try:
        manager = PostgresConnectionManager(config)
        manager.execute_sync(tables_sql)
        logger.info("PostgreSQL tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL tables: {e}")
        return False


if __name__ == "__main__":
    async def main():
        success = await test_connection()
        print(f"Connection test: {'SUCCESS' if success else 'FAILED'}")

        if success:
            create_tables_sync()
            print("Tables created successfully")

    asyncio.run(main())
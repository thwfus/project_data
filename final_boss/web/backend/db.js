
import mysql from "mysql2/promise";
import dotenv from "dotenv";

dotenv.config();

const pool = mysql.createPool({
  host: process.env.DB_HOST,
  port: Number(process.env.DB_PORT || 3306),
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

// test kết nối thật
(async () => {
  try {
    const conn = await pool.getConnection();
    await conn.query("SELECT 1");
    conn.release();
    console.log("✅ Connected to MySQL:", process.env.DB_NAME);
  } catch (err) {
    console.error("❌ MySQL connection failed:", err.message);
  }
})();

export default pool;

// web/frontend/src/component/Header.jsx

import React, { useEffect, useState } from "react";
import styles from "../styles/Header.module.css";
import { Link, useNavigate } from "react-router-dom";

const API_BASE = "http://localhost:3000";

function Header() {
  const navigate = useNavigate();

  // Lấy user sẵn từ localStorage để render ngay (nếu có)
  const [user, setUser] = useState(() => {
    try {
      const raw = localStorage.getItem("user");
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  });

  // Khi refresh trang: nếu có token thì gọi /api/me để xác nhận + lấy user mới nhất
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;

    fetch(`${API_BASE}/api/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(async (r) => {
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d?.message || "Token không hợp lệ");
        if (!d?.user) throw new Error("Không có dữ liệu user");
        localStorage.setItem("user", JSON.stringify(d.user));
        setUser(d.user);
      })
      .catch(() => {
        // token lỗi / hết hạn
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        setUser(null);
      });
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setUser(null);
    navigate("/login"); // hoặc navigate("/")
  };

  return (
    <header className={styles.header}>
      <div className={styles.headerContentWrapper}>
        <div className={styles.logoContainer}>
          <Link to="/">
            <img 
              src="public/logo_nhom3.png" // Bạn thay đường dẫn ảnh logo của bạn vào đây
              alt="Diabetes Logo" 
              className={styles.logoImage} 
            />
          </Link>
        </div>

        <div className={styles.searchContainer}>
          <input
            type="text"
            placeholder="Tìm kiếm 🔍"
            className={styles.searchInput}
          />
        </div>

        <div className={styles.authContainer}>
          {user ? (
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <span>
                Xin chào, <b>{user.username}</b>
              </span>

              <button className={styles.authButton} onClick={handleLogout}>
                Đăng xuất
              </button>
            </div>
          ) : (
            <>
              {/* Bạn có thể dùng Link hoặc navigate đều được */}
              <Link to="/register">
                <button className={styles.authButton}>Đăng ký</button>
              </Link>

              <Link to="/login">
                <button className={styles.authButton}>Đăng nhập</button>
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;

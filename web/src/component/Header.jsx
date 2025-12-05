// D:\Python\web\component\Header.jsx

import React from 'react';
import styles from '../styles/Header.module.css'; 

function Header() {
  return (
    <header className={styles.header}>
      {/* Container Wrapper để căn giữa nội dung */}
      <div className={styles.headerContentWrapper}>
        <div className={styles.leftSpace}></div>

        <div className={styles.searchContainer}>
            <input type="text" placeholder="Tìm kiếm 🔍" className={styles.searchInput} />
        </div>

        <div className={styles.authContainer}>
            <a href="Register.jsx"><button className={styles.authButton}>Đăng ký</button></a>
            <a href="Log_in.jsx"><button className={styles.authButton}>Đăng nhập</button></a>
        </div>
      </div>
    </header>
  );
}

export default Header;
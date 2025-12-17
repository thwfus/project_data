import React from 'react';
import { NavLink } from "react-router-dom";
import styles from '../styles/Navbar.module.css';

function Navbar() {
  return (
    <nav className={styles.navBar}>
      <div className={styles.navContentWrapper}>
        <div className={styles.navLinks}>
          <NavLink 
            to="/" 
            className={({ isActive }) => isActive ? styles.active : ""}>
            Trang Chủ
          </NavLink>

          <NavLink to="/info" className={({ isActive }) => isActive ? styles.active : ""}> Thông tin</NavLink>
          <NavLink to="/predict" className={({ isActive }) => isActive ? styles.active : ""}>Dự đoán</NavLink>
          <NavLink to="/history" className={({ isActive }) => isActive ? styles.active : ""}>Lịch sử </NavLink>
        </div>

        {/* Phần chuyển ngôn ngữ */}
        <div className={styles.langWrapper} title="Chuyển ngôn ngữ">
          <img 
            src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Flag_of_Vietnam.svg/1200px-Flag_of_Vietnam.svg.png" 
            alt="Vietnamese Flag" 
            className={styles.flagIcon} 
          />
          <span className={styles.langText}>VI</span>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
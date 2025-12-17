import React from 'react';
import { Link } from 'react-router-dom';// Nếu bạn dùng Next.js, hoặc dùng <a> nếu dùng React thuần
import styles from '../styles/HomePage.module.css';

function Hero() {
  return (
    <section className={styles.hero}>
      <h1>Dự đoán nguy cơ tiểu đường bằng trí tuệ nhân tạo</h1>
      <p>Sử dụng mô hình máy học tiên tiến để đánh giá sức khỏe của bạn chỉ trong vài phút.</p>
      <Link to="/predict">
        <button className={styles.ctaButton}>Bắt đầu kiểm tra ngay</button>
      </Link>
    </section>
  );
}

function CardContainer() {
  const features = [
    { title: "Phân tích AI", desc: "Dựa trên tập dữ liệu y tế lớn với độ chính xác cao.", icon: "🤖" },
    { title: "Bảo mật", desc: "Thông tin sức khỏe của bạn được mã hóa và bảo mật tuyệt đối.", icon: "🔒" },
    { title: "Gợi ý hữu ích", desc: "Nhận lời khuyên về lối sống dựa trên kết quả dự đoán.", icon: "🥗" }
  ];

  return (
    <div className={styles.cardContainer}>
      {features.map((item, index) => (
        <div key={index} className={styles.card}>
          <div className={styles.icon}>{item.icon}</div>
          <h3>{item.title}</h3>
          <p>{item.desc}</p>
        </div>
      ))}
    </div>
  );
}

function HomePage() {
  return (
    <div className={styles.pageWrapper}>
      <main className={styles.mainContent}>
        <Hero />
        <h2 style={{ textAlign: 'center', marginTop: '40px' }}>Tại sao nên chọn chúng tôi?</h2>
        <CardContainer />
      </main>
    </div>
  );
}

export default HomePage;
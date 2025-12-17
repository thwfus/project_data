import React from "react";
import styles from "../styles/Info.module.css";

function Info() {
  const infoData = [
    { title: "Thống kê", content: "Khoảng 589 triệu người trên thế giới đang sống chung với tiểu đường (2024).", icon: "📊" },
    { title: "Số ca tử vong", content: "Ước tính có khoảng 3,4 triệu người tử vong do các bệnh liên quan đến đái tháo đường trên toàn thế giới, tương đương cứ mỗi 9 giây lại có một người qua đời vì căn bệnh này (2024).", icon: "☠️" },
    { title: "Triệu chứng", content: "Khát nước liên tục, đi tiểu nhiều lần và sụt cân không rõ nguyên nhân.", icon: "⚠️" },
    { title: "Chế độ ăn", content: "Ưu tiên thực phẩm giàu chất xơ, ngũ cốc nguyên hạt và hạn chế đường.", icon: "🥗" },
    { title: "Vận động", content: "Duy trì ít nhất 30 phút thể dục mỗi ngày giúp kiểm soát đường huyết tốt hơn.", icon: "🏃" },
    { title: "Biến chứng", content: "Có thể gây ảnh hưởng đến tim mạch, mắt, thận và hệ thần kinh.", icon: "🏥" },
    { title: "Phòng ngừa", content: "Kiểm soát cân nặng và khám sức khỏe định kỳ là chìa khóa vàng.", icon: "🛡️" }
  ];

  return (
    <div className={styles.pageWrapper}>
      <main className={styles.mainContent}>
        <h2 className={styles.pageTitle}>Thông tin về bệnh tiểu đường</h2>
        <p className={styles.subTitle}>Những kiến thức cơ bản và số liệu cập nhật bạn cần biết</p>
        
        <div className={styles.infoGrid}>
          {infoData.map((item, index) => (
            <div key={index} className={styles.infoBox}>
              <div className={styles.icon}>{item.icon}</div>
              <h3>{item.title}</h3>
              <p>{item.content}</p>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}

export default Info;
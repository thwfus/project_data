export default function Suggestions({ form, probability }) {
  const p = Number(probability);

  if (!Number.isFinite(p)) return null;

  const tips = [];

  // ====== Các biến từ form ======
  const bmi = form?.bmi === "" ? null : Number(form?.bmi);
  const highbp = form?.highbp;
  const highchol = form?.highchol;
  const smoker = form?.smoker;
  const phys = form?.physactivity;
  const fruits = form?.fruits;
  const veggies = form?.veggies;
  const alcohol = form?.hvyalcoholconsump;
  const healthcare = form?.anyhealthcare;
  const nodoc = form?.nodocbcost;
  const genhlth = form?.genhlth;
  const menthlth = form?.menthlth;
  const physhlth = form?.physhlth;
  const diffwalk = form?.diffwalk;
  const age = form?.age_real;
  const sex = form?.sex;

  // ====== Gợi ý theo xác suất AI ======
  if (p >= 0.6) {
    tips.push("🔴 Nguy cơ cao: nên đi khám sớm và kiểm tra đường huyết định kỳ.");
  } else if (p >= 0.3) {
    tips.push("🟠 Nguy cơ trung bình: cần điều chỉnh lối sống để giảm nguy cơ.");
  } else {
    tips.push("🟢 Nguy cơ thấp: tiếp tục duy trì lối sống lành mạnh.");
  }

  // ====== BMI ======
  if (bmi !== null) {
    if (bmi >= 30) {
      tips.push("⚠️ BMI rất cao: nên có kế hoạch giảm cân có kiểm soát.");
    } else if (bmi >= 25) {
      tips.push("⚠️ BMI cao: ưu tiên ăn ít tinh bột nhanh, tăng vận động.");
    } else if (bmi < 18.5) {
      tips.push("⚠️ BMI thấp: cần đảm bảo dinh dưỡng đầy đủ.");
    }
  }

  // ====== Huyết áp & cholesterol ======
  if (highbp === "1")
    tips.push("❤️ Huyết áp cao: giảm muối, ngủ đủ, kiểm soát stress.");

  if (highchol === "1")
    tips.push("🩸 Cholesterol cao: hạn chế đồ chiên, tăng chất xơ.");

  // ====== Hút thuốc & rượu ======
  if (smoker === "1")
    tips.push("🚭 Hút thuốc làm tăng nguy cơ tiểu đường và tim mạch – nên cai.");

  if (alcohol === "1")
    tips.push("🍺 Uống nhiều rượu bia: nên giảm để bảo vệ gan và chuyển hóa.");

  // ====== Vận động ======
  if (phys === "0")
    tips.push("🏃 Ít vận động: nên tập ≥150 phút/tuần (đi bộ nhanh, đạp xe).");

  if (diffwalk === "1")
    tips.push("🦵 Khó đi lại: nên chọn bài tập nhẹ (yoga, vật lý trị liệu).");

  // ====== Rau củ & trái cây ======
  if (fruits === "0")
    tips.push("🍎 Ít ăn trái cây: nên bổ sung trái cây tươi mỗi ngày.");

  if (veggies === "0")
    tips.push("🥬 Ít ăn rau: tăng rau xanh để cải thiện chuyển hóa.");

  // ====== Sức khỏe tổng quát ======
  if (genhlth && Number(genhlth) >= 4)
    tips.push("📋 Sức khỏe tổng quát kém: nên kiểm tra sức khỏe định kỳ.");

  if (menthlth && Number(menthlth) >= 15)
    tips.push("🧠 Sức khỏe tinh thần kém: nên nghỉ ngơi và giảm stress.");

  if (physhlth && Number(physhlth) >= 15)
    tips.push("💪 Sức khỏe thể chất kém: nên theo dõi và tập phục hồi.");

  // ====== Tiếp cận y tế ======
  if (healthcare === "0")
    tips.push("🏥 Không có bảo hiểm y tế: nên cân nhắc đăng ký BHYT.");

  if (nodoc === "1")
    tips.push("💰 Tránh đi khám vì chi phí: nên tìm cơ sở y tế công.");

  // ====== Tuổi & giới ======
  if (age && Number(age) >= 45)
    tips.push("📅 Tuổi ≥45: nên tầm soát tiểu đường định kỳ.");

  if (sex === "1" && age && Number(age) >= 40)
    tips.push("👨 Nam trung niên: chú ý cân nặng và huyết áp.");

  // ====== Render ======
  return (
    <div
      style={{
        marginTop: 18,
        padding: 14,
        border: "1px solid #ddd",
        borderRadius: 8,
        background: "#fafafa",
      }}
    >
      <h3 style={{ marginTop: 0 }}>Gợi ý cải thiện sức khỏe</h3>
      <ul style={{ margin: 0, paddingLeft: 18 }}>
        {tips.map((t, i) => (
          <li key={i} style={{ marginBottom: 6 }}>
            {t}
          </li>
        ))}
      </ul>
      <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>
        * Gợi ý mang tính tham khảo, không thay thế tư vấn y tế.
      </div>
    </div>
  );
}

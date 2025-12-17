import { useMemo, useState } from "react";
import Suggestions from "./Suggestions";  
import styles from "../styles/Predict.module.css";

const BIN = [
  { value: "", label: "(bỏ lọc)" },
  { value: "0", label: "0 - Không" },
  { value: "1", label: "1 - Có" },
];

const FIELDS = [
  { name: "sex", label: "Giới tính", type: "select", options: [
    { value: "", label: "(bỏ lọc)" },
    { value: "0", label: "0 - Nữ" },
    { value: "1", label: "1 - Nam" },
  ]},
  { name: "age_real", label: "Tuổi", type: "input", placeholder: "vd: 21" },

  { name: "education", label: "Trình độ học vấn", type: "select", options: [
    { value: "", label: "(bỏ lọc)" },
    { value: "1", label: "1 - Chưa từng đi học hoặc chỉ học mẫu giáo" },
    { value: "2", label: "2 - Tiểu học" },
    { value: "3", label: "3 - Chưa tốt nghiệp THPT" },
    { value: "4", label: "4 - Tốt nghiệp THPT" },
    { value: "5", label: "5 - Học 1-3 năm cao đẳng/đại học hoặc trường kỹ thuật" },
    { value: "6", label: "6 - Tốt nghiệp đại học" },
  ]},
  { name: "income", label: "Thu nhập hộ gia đình", type: "select", options: [
    { value: "", label: "(bỏ lọc)" },
    { value: "1", label: "1 - Dưới 10.000 USD" },
    { value: "2", label: "2 - 10.000-15.000 USD" },
    { value: "3", label: "3 - 15.000-20.000 USD" },
    { value: "4", label: "4 - 20.000-25.000 USD" },
    { value: "5", label: "5 - 25.000-35.000 USD" },
    { value: "6", label: "6 - 35.000-50.000 USD" },
    { value: "7", label: "7 - 50.000-75.000 USD" },
    { value: "8", label: "8 - Trên 75.000 USD" },
  ] },

  { name: "bmi", label: "Chỉ số cơ thể (kg/m²)", type: "input", placeholder: "23" },
  { name: "highbp", label: "Từng được chẩn đoán huyết áp cao", type: "select", options: BIN },
  { name: "highchol", label: "Từng được chẩn đoán cholesterol cao", type: "select", options: BIN },
  { name: "cholcheck", label: "Đã kiểm tra cholesterol < 5 năm", type: "select", options: BIN },

  { name: "smoker", label: "Đã hút 100 điếu thuốc", type: "select", options: BIN },
  { name: "stroke", label: "Từng bị đột quỵ", type: "select", options: BIN },
  { name: "heartdiseaseorattack", label: "Từng mắc bệnh tim hay nhồi máu cơ tim", type: "select", options: BIN },
  { name: "physactivity", label: "Có hoạt động thể chất < 30 ngày", type: "select", options: BIN },
  { name: "fruits", label: "Ăn trái cây ít nhất 1 lần/ngày", type: "select", options: BIN },
  { name: "veggies", label: "Ăn rau ít nhất 1 lần/ngày", type: "select", options: BIN },
  { name: "hvyalcoholconsump", label: "Uống rượu mạnh", type: "select", options: BIN },
  { name: "anyhealthcare", label: "Có bảo hiểm y tế", type: "select", options: BIN },
  { name: "nodocbccost", label: "Không đi khám bác sĩ trong 12 tháng qua", type: "select", options: BIN },

  { name: "genhlth", label: "Đánh giá chủ quan sức khỏe cơ thể", type: "input", placeholder: "1..5" },
  { name: "menthlth", label: "Số ngày sức khỏe tinh thần không tốt trong tháng", type: "input", placeholder: "0..30" },
  { name: "physhlth", label: "Số ngày sức khỏe thể chất không tốt trong tháng", type: "input", placeholder: "0..30" },
  { name: "diffwalk", label: "Khó khăn khi đi lại", type: "select", options: BIN },
];

const GROUPS = [
  {
    title: "Thông tin cá nhân",
    fields: ["sex", "age_real", "education", "income"],
  },
  {
    title: "Chỉ số sức khỏe",
    fields: ["bmi", "highbp", "highchol", "stroke", "genhlth", "menthlth", "physhlth", "heartdiseaseorattack"],
  },
  {
    title: "Thói quen sinh hoạt",
    fields: ["smoker", "physactivity", "fruits", "veggies", "hvyalcoholconsump", "diffwalk"],
  },
  {
    title: "Tiếp cận y tế",
    fields: ["anyhealthcare", "nodocbccost", "cholcheck"],
  },
];

function ageToCode(age) {
  const a = Number(age);
  if (!Number.isFinite(a)) return "";
  if (a < 18) return ""; 
  if (a <= 24) return 1;
  if (a <= 29) return 2;
  if (a <= 34) return 3;
  if (a <= 39) return 4;
  if (a <= 44) return 5;
  if (a <= 49) return 6;
  if (a <= 54) return 7;
  if (a <= 59) return 8;
  if (a <= 64) return 9;
  if (a <= 69) return 10;
  if (a <= 74) return 11;
  if (a <= 79) return 12;
  return 13; // >= 80
}

export default function Predict() {
  const [form, setForm] = useState(() => {
  const init = {};
  FIELDS.forEach(f => init[f.name] = "");
  init.age_real = "";  
  init.sex = "1";
  init.bmi = "31";
  init.highbp = "1";
  return init;
});


  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const qs = useMemo(() => {
  const p = new URLSearchParams();

  Object.entries(form).forEach(([k, v]) => {
    if (k === "age_real") return; // không gửi tuổi thật
    if (v !== "" && v !== null && v !== undefined) p.set(k, v);
  });

  if (form.age_real !== "") {
    const code = ageToCode(form.age_real);
    if (code !== "") p.set("age", String(code));
  }

  return p.toString();
}, [form]);



  const onChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      // GỌI AI
      const res = await fetch(`/api/diabetes/risk-ai?${qs}`);
      const data = await res.json();

      console.log("AI response =", data);
console.log("QS sent =", qs);


const probability = data?.ai_probability; 
console.log("prob=", probability, "token=", localStorage.getItem("token"));
if (typeof probability === "number") {
  const token = localStorage.getItem("token");
  if (token) {
    await fetch("http://localhost:3000/api/history", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
  // ====== 21 biến ======
  sex: form.sex === "" ? null : Number(form.sex),

  // age: backend muốn code 1..13
  age: form.age_real === "" ? null : Number(ageToCode(form.age_real)),

  bmi: form.bmi === "" ? null : Number(form.bmi),
  highbp: form.highbp === "" ? null : Number(form.highbp),
  highchol: form.highchol === "" ? null : Number(form.highchol),
  cholcheck: form.cholcheck === "" ? null : Number(form.cholcheck),

  smoker: form.smoker === "" ? null : Number(form.smoker),
  stroke: form.stroke === "" ? null : Number(form.stroke),
  heartdiseaseorattack:
    form.heartdiseaseorattack === "" ? null : Number(form.heartdiseaseorattack),

  physactivity: form.physactivity === "" ? null : Number(form.physactivity),
  fruits: form.fruits === "" ? null : Number(form.fruits),
  veggies: form.veggies === "" ? null : Number(form.veggies),
  hvyalcoholconsump:
    form.hvyalcoholconsump === "" ? null : Number(form.hvyalcoholconsump),

  anyhealthcare: form.anyhealthcare === "" ? null : Number(form.anyhealthcare),
  nodocbccost: form.nodocbccost === "" ? null : Number(form.nodocbccost),

  genhlth: form.genhlth === "" ? null : Number(form.genhlth),
  menthlth: form.menthlth === "" ? null : Number(form.menthlth),
  physhlth: form.physhlth === "" ? null : Number(form.physhlth),

  diffwalk: form.diffwalk === "" ? null : Number(form.diffwalk),

  education: form.education === "" ? null : Number(form.education),
  income: form.income === "" ? null : Number(form.income),

  // ====== kết quả ======
  risk_prob: probability,
  risk_label: probability >= 0.6 ? "High" : probability >= 0.3 ? "Medium" : "Low",
}),

    });
  }
}




      if (!res.ok) throw new Error(data?.error || data?.message || "Request failed");
      setResult(data);
    } catch (err) {
      setError(err.message || "Error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.pageWrapper}>
      <div className={styles.container}>
        <h2 className={styles.title}>Dự đoán nguy cơ mắc bệnh tiểu đường</h2>
        <p className={styles.subTitle}>Sử dụng trí tuệ nhân tạo để đưa ra khả năng bệnh</p>
        <form onSubmit={onSubmit} className={styles.form}>
          {GROUPS.map((group) => (
            <fieldset key={group.title} className={styles.fieldset}>
              <legend className={styles.legend}>{group.title}</legend>
              <div className={styles.grid}>
                {FIELDS.filter((f) => group.fields.includes(f.name)).map((f) => (
                  <div key={f.name} className={styles.fieldItem}>
                    <label className={styles.label}>{f.label}</label>
                    {f.type === "select" ? (
                      <select name={f.name} value={form[f.name]} onChange={onChange} className={styles.select}>
                        {f.options.map((op) => (
                          <option key={op.value} value={op.value}>{op.label}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        name={f.name}
                        className={styles.input}
                        value={form[f.name]}
                        onChange={onChange}
                        placeholder={f.placeholder || ""}
                      />
                    )}
                  </div>
                ))}
              </div>
            </fieldset>
          ))}

          <button type="submit" disabled={loading} className={styles.submitBtn}>
            {loading ? "Đang tính..." : "Tính toán kết quả"}
          </button>

          </form>

          <div style={{ marginTop: 12, opacity: 0.75 }}>
            URL gọi: <code>/api/diabetes/risk-ai?{qs}</code>
          </div>

          {error && <div style={{ marginTop: 12 }}>❌ {error}</div>}

          {result && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 18 }}>
                <b>Nguy cơ mắc tiểu đường (AI):</b>
                <Suggestions form={form} probability={result?.ai_probability} />

              </div>

              <div style={{ fontSize: 32, fontWeight: "bold", marginTop: 6 }}>
                {result.ai_percent ?? "null"} %
              </div>

              <div style={{ opacity: 0.75, marginTop: 6 }}>
                Xác suất (ai_probability):{" "}
                {result.ai_probability === null || result.ai_probability === undefined
                  ? "null"
                  : Number(result.ai_probability).toFixed(4)}
              </div>

              <details style={{ marginTop: 10 }}>
                <summary>Debug JSON</summary>
                <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </div>
          )}
        </div>
    </div>
  );
}

import React, { useEffect, useMemo, useState } from "react";
import styles from "../styles/History.module.css";

const API_BASE = "http://localhost:3000";

function fmtDate(s) {
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s;
  }
}

function labelFromProb(p) {
  if (p === null || p === undefined || Number.isNaN(Number(p))) return "";
  const x = Number(p);
  if (x >= 0.6) return "High";
  if (x >= 0.3) return "Medium";
  return "Low";
}

export default function HistoryPage() {
  
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  const token = useMemo(() => localStorage.getItem("token"), []);

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      setErr("");

      if (!token) {
        setErr("Bạn chưa đăng nhập. Hãy đăng nhập để xem lịch sử.");
        setLoading(false);
        return;
      }

      try {
        const r = await fetch(`${API_BASE}/api/history`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data?.message || "Fetch history failed");
        setRows(Array.isArray(data) ? data : []);
      } catch (e) {
        setErr(String(e.message || e));
      } finally {
        setLoading(false);
      }
    };

    run();
  }, [token]);

  const thStyle = {
    textAlign: "left",
    borderBottom: "2px solid #eee",
    padding: "10px 8px",
    whiteSpace: "nowrap",
    fontWeight: 700,
    fontSize: 13,
  };

  const tdStyle = {
    padding: "10px 8px",
    borderBottom: "1px solid #f2f2f2",
    whiteSpace: "nowrap",
    fontSize: 13,
  };

  const headers = [
    "Thời gian",
    "Xác suất",
    "Mức",
    "Sex",
    "Age",
    "BMI",
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Education",
    "Income",
  ];

  return (
    <div className={styles.pageWrapper}>
      <div className={styles.container}>
        <h2 className={styles.title}>Lịch sử dự đoán</h2>
      
        <p className={styles.description}>
          Danh sách tối đa 100 lần dự đoán gần nhất (theo tài khoản đang đăng nhập).
        </p>

        {loading && <div className={styles.statusText}>Đang tải...</div>}

        {!loading && err && (
          <div className={styles.errorBox}>
            <b>Lỗi:</b> {err}
          </div>
        )}

        {!loading && !err && rows.length === 0 && (
          <div className={styles.emptyBox}>
            Chưa có lịch sử nào. Hãy sang trang <b>Dự đoán</b> và bấm “Tính nguy cơ”.
          </div>
        )}

        {!loading && !err && rows.length > 0 && (
          <div className={styles.tableResponsive}>
            <table className={styles.table}>
              <thead>
                <tr>
                  {headers.map((h) => (
                    <th key={h} style={thStyle}>
                      {h}
                    </th>
                  ))}
                  <th style={thStyle}>Hành động</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const prob = Number(r.risk_prob);
                  const probText = Number.isFinite(prob)
                    ? prob <= 1
                      ? `${Math.round(prob * 100)}%`
                      : `${Math.round(prob)}%`
                    : "";

                  const label = r.risk_label || labelFromProb(prob);
                  
                  // Xác định màu cho mức độ
                  const statusClass = 
                    label === "High" ? styles.high : 
                    label === "Medium" ? styles.medium : styles.low;

                  return (
                    <tr key={r.id}>
                      <td style={tdStyle}>{fmtDate(r.created_at)}</td>
                      <td style={tdStyle}><b>{probText}</b></td>
                      <td style={tdStyle}>
                        <span className={`${styles.statusBadge} ${statusClass}`}>
                          {label}
                        </span>
                      </td>

                      {/* 21 biến */}
                      <td style={tdStyle}>{r.sex ?? ""}</td>
                      <td style={tdStyle}>{r.age ?? ""}</td>
                      <td style={tdStyle}>{r.bmi ?? ""}</td>
                      <td style={tdStyle}>{r.highbp ?? ""}</td>
                      <td style={tdStyle}>{r.highchol ?? ""}</td>
                      <td style={tdStyle}>{r.cholcheck ?? ""}</td>
                      <td style={tdStyle}>{r.smoker ?? ""}</td>
                      <td style={tdStyle}>{r.stroke ?? ""}</td>
                      <td style={tdStyle}>{r.heartdiseaseorattack ?? ""}</td>
                      <td style={tdStyle}>{r.physactivity ?? ""}</td>
                      <td style={tdStyle}>{r.fruits ?? ""}</td>
                      <td style={tdStyle}>{r.veggies ?? ""}</td>
                      <td style={tdStyle}>{r.hvyalcoholconsump ?? ""}</td>
                      <td style={tdStyle}>{r.anyhealthcare ?? ""}</td>
                      <td style={tdStyle}>{r.nodocbccost ?? ""}</td>
                      <td style={tdStyle}>{r.genhlth ?? ""}</td>
                      <td style={tdStyle}>{r.menthlth ?? ""}</td>
                      <td style={tdStyle}>{r.physhlth ?? ""}</td>
                      <td style={tdStyle}>{r.diffwalk ?? ""}</td>
                      <td style={tdStyle}>{r.education ?? ""}</td>
                      <td style={tdStyle}>{r.income ?? ""}</td>
                      <td style={tdStyle}>
                        <button
                          className={styles.deleteRowBtn}
                          onClick={async () => {
                            if (!confirm("Xóa lần dự đoán này?")) return;
                            const rDel = await fetch(`http://localhost:3000/api/history/${r.id}`, {
                              method: "DELETE",
                              headers: { Authorization: `Bearer ${token}` },
                            });
                            const j = await rDel.json().catch(() => ({}));
                            if (!rDel.ok) {
                              alert(j.message || "Xóa thất bại");
                              return;
                            }
                            setRows((prev) => prev.filter((x) => x.id !== r.id));
                          }}
                        >
                          Xóa
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        <button 
          className={styles.deleteAllBtn}
          onClick={async () => {
            if (!confirm("Xóa toàn bộ lịch sử?")) return;
            const rDel = await fetch("http://localhost:3000/api/history", {
              method: "DELETE",
              headers: { Authorization: `Bearer ${token}` },
            });
            const j = await rDel.json().catch(() => ({}));
            if (!rDel.ok) return alert(j.message || "Xóa thất bại");
            setRows([]);
          }}
        >
          Xóa hết
        </button>
      </div>
    </div>
  );
}



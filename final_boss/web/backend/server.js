import express from "express";
import cors from "cors";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import dotenv from "dotenv";
import pool from "./db.js";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT || 3000);
const JWT_SECRET = process.env.JWT_SECRET || "secret_demo";

app.use(cors({ origin: "http://localhost:5173", credentials: true }));
app.use(express.json());

// ===================== AUTH =====================

// Đăng ký
app.post("/api/register", async (req, res) => {
  try {
    const { username, email, password } = req.body;

    if (!username || !email || !password) {
      return res.status(400).json({ message: "Thiếu thông tin" });
    }

    const [rows] = await pool.query(
      "SELECT id FROM users WHERE username = ? OR email = ?",
      [username, email]
    );
    if (rows.length > 0) {
      return res.status(400).json({ message: "Username hoặc email đã tồn tại" });
    }

    const passwordHash = await bcrypt.hash(password, 10);

    await pool.query(
      "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
      [username, email, passwordHash]
    );

    return res.status(201).json({ message: "Đăng ký thành công" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Lỗi server" });
  }
});

// Đăng nhập
app.post("/api/login", async (req, res) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      return res.status(400).json({ message: "Thiếu username hoặc password" });
    }

    const [rows] = await pool.query(
      "SELECT id, username, email, password_hash FROM users WHERE username = ?",
      [username]
    );

    if (rows.length === 0) {
      return res.status(400).json({ message: "Sai username hoặc password" });
    }

    const user = rows[0];
    const ok = await bcrypt.compare(password, user.password_hash);
    if (!ok) {
      return res.status(400).json({ message: "Sai username hoặc password" });
    }

    const token = jwt.sign(
      { id: user.id, username: user.username, email: user.email },
      JWT_SECRET,
      { expiresIn: "1h" }
    );

    res.json({ message: "Đăng nhập thành công", token });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Lỗi server" });
  }
});

// Lấy thông tin user từ token
app.get("/api/me", async (req, res) => {
  try {
    const auth = req.headers.authorization;
    if (!auth?.startsWith("Bearer ")) {
      return res.status(401).json({ message: "Thiếu token" });
    }

    const token = auth.split(" ")[1];
    const payload = jwt.verify(token, JWT_SECRET);
    res.json({ user: payload });
  } catch (err) {
    return res.status(401).json({ message: "Token không hợp lệ" });
  }
});

// ===================== BASIC =====================

app.get("/", (req, res) => res.send("Backend OK"));

app.get("/test-db", async (req, res) => {
  try {
    const [rows] = await pool.query("SELECT 1 AS ok");
    res.json({ message: "Kết nối database OK", rows });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Kết nối database FAILED", error: err.message });
  }
});

// ===================== DIABETES APIs =====================

app.get("/api/diabetes/by-sex", async (req, res) => {
  try {
    const sql = `
      SELECT 
        p.Sex AS sex_code,
        COUNT(*) AS total,
        SUM(f.Diabetes_binary) AS diabetes_cases,
        AVG(f.Diabetes_binary) AS diabetes_rate
      FROM fact_diabetes f
      JOIN dim_person p ON f.PersonID = p.PersonID
      GROUP BY p.Sex
      ORDER BY p.Sex
    `;
    const [rows] = await pool.query(sql);

    const out = rows.map((r) => ({
      sex_code: r.sex_code,
      sex: String(r.sex_code) === "1" ? "Male" : "Female",
      total: Number(r.total),
      diabetes_cases: Number(r.diabetes_cases),
      diabetes_rate: Number(r.diabetes_rate),
    }));

    res.json(out);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Query failed", error: err.message });
  }
});

app.get("/api/diabetes/by-bmi", async (req, res) => {
  try {
    const sql = `
      SELECT
        CASE
          WHEN h.BMI < 18.5 THEN '<18.5'
          WHEN h.BMI < 25 THEN '18.5-24.9'
          WHEN h.BMI < 30 THEN '25-29.9'
          ELSE '>=30'
        END AS bmi_group,
        COUNT(*) AS total,
        AVG(f.Diabetes_binary) AS diabetes_rate
      FROM fact_diabetes f
      JOIN dim_healthstatus h ON f.HealthID = h.HealthID
      GROUP BY bmi_group
      ORDER BY
        CASE bmi_group
          WHEN '<18.5' THEN 1
          WHEN '18.5-24.9' THEN 2
          WHEN '25-29.9' THEN 3
          ELSE 4
        END
    `;
    const [rows] = await pool.query(sql);

    res.json(
      rows.map((r) => ({
        bmi_group: r.bmi_group,
        total: Number(r.total),
        diabetes_rate: Number(r.diabetes_rate),
      }))
    );
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Query failed", error: err.message });
  }
});

app.get("/api/diabetes/by-lifestyle", async (req, res) => {
  try {
    const sql = `
      SELECT
        l.Smoker AS smoker,
        l.PhysActivity AS phys_activity,
        COUNT(*) AS total,
        AVG(f.Diabetes_binary) AS diabetes_rate
      FROM fact_diabetes f
      JOIN dim_lifestyle l ON f.LifeStyleID = l.LifeStyleID
      GROUP BY l.Smoker, l.PhysActivity
      ORDER BY l.Smoker, l.PhysActivity
    `;
    const [rows] = await pool.query(sql);

    res.json(
      rows.map((r) => ({
        smoker: Number(r.smoker),
        phys_activity: Number(r.phys_activity),
        total: Number(r.total),
        diabetes_rate: Number(r.diabetes_rate),
      }))
    );
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Query failed", error: err.message });
  }
});

// Giữ 1 bản risk (bản đầy đủ: join 4 dim)
app.get("/api/diabetes/risk", async (req, res) => {
  try {
    const {
      sex,
      age,
      education,
      income,
      bmi,
      highbp,
      highchol,
      smoker,
      physactivity,
      fruits,
      veggies,
      anyhealthcare,
      nodocbcost,
      cholcheck,
      diffwalk,
      genhlth,
      menthlth,
      physhlth,
      stroke,
      heartdiseaseorattack,
    } = req.query;

    const conditions = [];
    const params = [];

    // dim_person
    if (sex !== undefined) { conditions.push("p.Sex = ?"); params.push(Number(sex)); }
    if (age !== undefined) { conditions.push("p.AgeGroup = ?"); params.push(String(age)); }
    if (education !== undefined) { conditions.push("p.Education = ?"); params.push(String(education)); }
    if (income !== undefined) { conditions.push("p.Income = ?"); params.push(String(income)); }

    // dim_healthstatus
    if (highbp !== undefined) { conditions.push("h.HighBP = ?"); params.push(Number(highbp)); }
    if (highchol !== undefined) { conditions.push("h.HighChol = ?"); params.push(Number(highchol)); }
    if (genhlth !== undefined) { conditions.push("h.GenHlth = ?"); params.push(Number(genhlth)); }
    if (menthlth !== undefined) { conditions.push("h.MentHlth = ?"); params.push(Number(menthlth)); }
    if (physhlth !== undefined) { conditions.push("h.PhysHlth = ?"); params.push(Number(physhlth)); }
    if (stroke !== undefined) { conditions.push("h.Stroke = ?"); params.push(Number(stroke)); }
    if (heartdiseaseorattack !== undefined) { conditions.push("h.HeartDiseaseorAttack = ?"); params.push(Number(heartdiseaseorattack)); }

    // BMI: ±1 để tránh matched=0
    if (bmi !== undefined && bmi !== "") {
      const bmiNum = Number(bmi);
      if (Number.isFinite(bmiNum)) {
        conditions.push("h.BMI BETWEEN ? AND ?");
        params.push(bmiNum - 1, bmiNum + 1);
      }
    }

    // dim_lifestyle
    if (smoker !== undefined) { conditions.push("l.Smoker = ?"); params.push(Number(smoker)); }
    if (physactivity !== undefined) { conditions.push("l.PhysActivity = ?"); params.push(Number(physactivity)); }
    if (fruits !== undefined) { conditions.push("l.Fruits = ?"); params.push(Number(fruits)); }
    if (veggies !== undefined) { conditions.push("l.Veggies = ?"); params.push(Number(veggies)); }
    if (diffwalk !== undefined) { conditions.push("l.DiffWalk = ?"); params.push(Number(diffwalk)); }

    // dim_healthcareaccess
    if (anyhealthcare !== undefined) { conditions.push("a.AnyHealthcare = ?"); params.push(Number(anyhealthcare)); }
    if (nodocbcost !== undefined) { conditions.push("a.NoDocbcost = ?"); params.push(Number(nodocbcost)); }
    if (cholcheck !== undefined) { conditions.push("a.CholCheck = ?"); params.push(Number(cholcheck)); }

    const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";

    const sql = `
      SELECT
        COUNT(*) AS matched,
        AVG(f.Diabetes_binary) AS diabetes_rate
      FROM fact_diabetes f
      JOIN dim_person p ON f.PersonID = p.PersonID
      JOIN dim_healthstatus h ON f.HealthID = h.HealthID
      JOIN dim_lifestyle l ON f.LifeStyleID = l.LifeStyleID
      JOIN dim_healthcareaccess a ON f.AccessID = a.AccessID
      ${where}
    `;

    const [rows] = await pool.query(sql, params);
    const matched = Number(rows?.[0]?.matched || 0);
    const rateRaw = rows?.[0]?.diabetes_rate;

    res.json({
      matched,
      diabetes_rate: rateRaw === null ? null : Number(rateRaw),
      diabetes_percent: rateRaw === null ? null : Math.round(Number(rateRaw) * 10000) / 100,
      used_filters: conditions,
      note:
        matched === 0
          ? "Không có mẫu giống điều kiện. Thử bỏ bớt điều kiện hoặc chỉ nhập vài field."
          : "Kết quả là tỉ lệ mắc (AVG Diabetes_binary) trên các mẫu trong DB thỏa điều kiện.",
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Risk API failed", error: err.message });
  }
});

app.get("/api/diabetes/risk-ai", async (req, res) => {
  try {
    const q = req.query;

    const payload = {
  HighBP: req.query.highbp !== undefined ? Number(req.query.highbp) : undefined,
  HighChol: req.query.highchol !== undefined ? Number(req.query.highchol) : undefined,
  CholCheck: req.query.cholcheck !== undefined ? Number(req.query.cholcheck) : undefined,
  BMI: req.query.bmi !== undefined ? Number(req.query.bmi) : undefined,
  Smoker: req.query.smoker !== undefined ? Number(req.query.smoker) : undefined,
  Stroke: req.query.stroke !== undefined ? Number(req.query.stroke) : undefined,
  HeartDiseaseorAttack:
    req.query.heartdiseaseorattack !== undefined ? Number(req.query.heartdiseaseorattack) : undefined,
  PhysActivity: req.query.physactivity !== undefined ? Number(req.query.physactivity) : undefined,
  Fruits: req.query.fruits !== undefined ? Number(req.query.fruits) : undefined,
  Veggies: req.query.veggies !== undefined ? Number(req.query.veggies) : undefined,
  HvyAlcoholConsump:
    req.query.hvyalcoholconsump !== undefined ? Number(req.query.hvyalcoholconsump) : undefined,
  AnyHealthcare: req.query.anyhealthcare !== undefined ? Number(req.query.anyhealthcare) : undefined,
  NoDocbcCost: req.query.nodocbccost !== undefined ? Number(req.query.nodocbccost) : undefined,
  GenHlth: req.query.genhlth !== undefined ? Number(req.query.genhlth) : undefined,
  MentHlth: req.query.menthlth !== undefined ? Number(req.query.menthlth) : undefined,
  PhysHlth: req.query.physhlth !== undefined ? Number(req.query.physhlth) : undefined,
  DiffWalk: req.query.diffwalk !== undefined ? Number(req.query.diffwalk) : undefined,

  Sex: req.query.sex !== undefined ? Number(req.query.sex) : 1,
  Age: req.query.age !== undefined ? Number(req.query.age) : undefined,
  Education: req.query.education !== undefined ? Number(req.query.education) : undefined,
  Income: req.query.income !== undefined ? Number(req.query.income) : undefined,
};


    // xóa key undefined để gửi gọn
    Object.keys(payload).forEach((k) => payload[k] === undefined && delete payload[k]);

    const r = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await r.text();
    if (!text) return res.status(502).json({ message: "Python returned empty response" });

    let data;
    try { data = JSON.parse(text); }
    catch { return res.status(502).json({ message: "Python returned non-JSON", raw: text }); }

    if (!r.ok) return res.status(r.status).json(data);
    return res.json(data);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ message: "risk-ai failed", error: err.message });
  }
});

// ===================== HISTORY (JWT protected) =====================

// Lưu lịch sử dự đoán (chỉ user đăng nhập)
app.post("/api/history", requireAuth, async (req, res) => {
  try {
    const userId = req.user.id;

    // Lấy đầy đủ 21 biến + kết quả
    const {
      sex, age, bmi,
      highbp, highchol, smoker, physactivity,

      cholcheck, stroke, heartdiseaseorattack,
      fruits, veggies, hvyalcoholconsump,
      anyhealthcare, nodocbccost,
      genhlth, menthlth, physhlth, diffwalk,
      education, income,

      risk_prob, risk_label,
    } = req.body;

    if (risk_prob === undefined || risk_prob === null) {
      return res.status(400).json({ message: "Thiếu risk_prob" });
    }

    await pool.query(
      `
      INSERT INTO prediction_history
      (
        user_id,
        sex, age, bmi, highbp, highchol, smoker, physactivity,

        cholcheck, stroke, heartdiseaseorattack,
        fruits, veggies, hvyalcoholconsump,
        anyhealthcare, nodocbccost,
        genhlth, menthlth, physhlth, diffwalk,
        education, income,

        risk_prob, risk_label
      )
      VALUES
      (
        ?, ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?,
        ?, ?,
        ?, ?
      )
      `,
      [
        userId,

        sex ?? null,
        age ?? null,
        bmi ?? null,
        highbp ?? null,
        highchol ?? null,
        smoker ?? null,
        physactivity ?? null,

        cholcheck ?? null,
        stroke ?? null,
        heartdiseaseorattack ?? null,
        fruits ?? null,
        veggies ?? null,
        hvyalcoholconsump ?? null,
        anyhealthcare ?? null,
        nodocbccost ?? null,
        genhlth ?? null,
        menthlth ?? null,
        physhlth ?? null,
        diffwalk ?? null,
        education ?? null,
        income ?? null,

        Number(risk_prob),
        risk_label ?? null,
      ]
    );

    res.status(201).json({ message: "Đã lưu lịch sử dự đoán" });
  } catch (err) {
    console.error(err);
    res.status(500).json({
      message: "Lưu history thất bại",
      error: err.message,
    });
  }
});


// Lấy lịch sử dự đoán của chính user (mới nhất trước)
app.get("/api/history", requireAuth, async (req, res) => {
  try {
    const userId = req.user.id;
    const [rows] = await pool.query(
      `
      SELECT id, sex, age, bmi, highbp, highchol, smoker, physactivity,
       cholcheck, stroke, heartdiseaseorattack, fruits, veggies, hvyalcoholconsump,
       anyhealthcare, nodocbccost, genhlth, menthlth, physhlth, diffwalk,
       education, income,
       risk_prob, risk_label, created_at
      FROM prediction_history
      WHERE user_id = ?
      ORDER BY created_at DESC
      LIMIT 100
      `,
      [userId]
    );
    res.json(rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Lấy history thất bại", error: err.message });
  }
});

// ===================== DEBUG =====================

// Tạo bảng users (đúng với register/login: password_hash)
app.post("/api/debug/init-users-table", async (req, res) => {
  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    res.json({ message: "✅ users table created" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});


// // ===================== HISTORY =====================

// // Lưu lịch sử dự đoán
// app.post("/api/history", async (req, res) => {
//   try {
//     const {
//       user_id,
//       sex,
//       age,
//       bmi,
//       highbp,
//       highchol,
//       smoker,
//       physactivity,
//       risk_prob,
//       risk_label,
//     } = req.body;

//     if (risk_prob === undefined) {
//       return res.status(400).json({ message: "Thiếu risk_prob" });
//     }

//     await pool.query(
//       `
//       INSERT INTO prediction_history
//       (user_id, sex, age, bmi, highbp, highchol, smoker, physactivity, risk_prob, risk_label)
//       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
//       `,
//       [
//         user_id ?? null,
//         sex ?? null,
//         age ?? null,
//         bmi ?? null,
//         highbp ?? null,
//         highchol ?? null,
//         smoker ?? null,
//         physactivity ?? null,
//         risk_prob,
//         risk_label ?? null,
//       ]
//     );

//     res.status(201).json({ message: "Đã lưu lịch sử dự đoán" });
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ message: "Lưu history thất bại", error: err.message });
//   }
// });


// // Lấy lịch sử dự đoán (mới nhất trước)
// app.get("/api/history", async (req, res) => {
//   try {
//     const { user_id } = req.query;

//     let sql = `
//       SELECT *
//       FROM prediction_history
//     `;
//     const params = [];

//     if (user_id) {
//       sql += " WHERE user_id = ?";
//       params.push(Number(user_id));
//     }

//     sql += " ORDER BY created_at DESC LIMIT 100";

//     const [rows] = await pool.query(sql, params);
//     res.json(rows);
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ message: "Lấy history thất bại", error: err.message });
//   }
// });



function requireAuth(req, res, next) {
  try {
    const auth = req.headers.authorization;
    if (!auth?.startsWith("Bearer ")) {
      return res.status(401).json({ message: "Thiếu token" });
    }
    const token = auth.split(" ")[1];
    const payload = jwt.verify(token, JWT_SECRET);
    req.user = payload; // {id, username, email}
    next();
  } catch (err) {
    return res.status(401).json({ message: "Token không hợp lệ" });
  }
}


app.delete("/api/history/:id", requireAuth, async (req, res) => {
  try {
    const userId = req.user.id;           // requireAuth phải gán req.user
    const id = Number(req.params.id);
    if (!Number.isFinite(id)) return res.status(400).json({ message: "Invalid id" });

    const [result] = await pool.query(
      "DELETE FROM prediction_history WHERE id = ? AND user_id = ?",
      [id, userId]
    );

    if (result.affectedRows === 0) {
      return res.status(404).json({ message: "Not found" });
    }

    res.json({ ok: true });
  } catch (e) {
    console.error(e);
    res.status(500).json({ message: "Delete failed" });
  }
});



app.delete("/api/history", requireAuth, async (req, res) => {
  try {
    const userId = req.user.id;
    const [result] = await pool.query(
      "DELETE FROM prediction_history WHERE user_id = ?",
      [userId]
    );
    res.json({ ok: true, deleted: result.affectedRows });
  } catch (e) {
    console.error(e);
    res.status(500).json({ message: "Delete all failed" });
  }
});



















app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server đang chạy ở http://0.0.0.0:${PORT}`);
});

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Login.css";
import { useAuth } from "../context/AuthContext";

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const res = await fetch("http://localhost:3000/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        alert(data.message || "Đăng nhập thất bại");
        return;
      }

      //  lưu token
localStorage.setItem("token", data.token);

// gọi /api/me để lấy user
const meRes = await fetch("http://localhost:3000/api/me", {
  headers: { Authorization: `Bearer ${data.token}` },
});

const meData = await meRes.json();

if (!meRes.ok) {
  alert(meData.message || "Không lấy được thông tin user");
  return;
}

localStorage.setItem("user", JSON.stringify(meData.user)); // {id, username, email}

login({
  token: data.token,
  user: meData.user,
});

alert("Đăng nhập thành công!");

navigate("/");

    } catch (err) {
      console.error(err);
      alert("Lỗi kết nối server");
    }
  };

  return (
    <div className="pageWrapper">
      <div className="login-container">
        <div className="login-box">
          <h2 className="login-title">Đăng nhập</h2>

          <form onSubmit={handleSubmit}>
            <input
              type="text"
              placeholder="Tên đăng nhập"
              className="login-input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />

            <input
              type="password"
              placeholder="Mật khẩu"
              className="login-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />

            <button type="submit" className="login-btn">
              Đăng nhập
            </button>
          </form>

          <div className="login-link">
            Chưa có tài khoản?{" "}
            <a href="#" onClick={() => navigate("/register")}>
              Đăng ký
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

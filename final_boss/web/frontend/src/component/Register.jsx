import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Register.css";

export default function Register() {
  const navigate = useNavigate();

  // state
  const [fullname, setFullname] = useState("");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const res = await fetch("http://localhost:3000/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username,
          email,
          password
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        alert(data.message || "Đăng ký thất bại");
        return;
      }

      alert("Đăng ký thành công!");
      navigate("/login");

    } catch (err) {
      console.error(err);
      alert("Lỗi kết nối server");
    }
  };

  return (
    <div className="pageWrapper">
      <div className="register-container">
        <div className="register-box">
          <h2 className="register-title">Đăng ký</h2>

          <form onSubmit={handleSubmit}>
            <input
              type="text"
              placeholder="Họ và tên"
              className="register-input"
              value={fullname}
              onChange={(e) => setFullname(e.target.value)}
            />

            <input
              type="email"
              placeholder="Email"
              className="register-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />

            <input
              type="text"
              placeholder="Tên đăng nhập"
              className="register-input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />

            <input
              type="password"
              placeholder="Mật khẩu"
              className="register-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />

            <button type="submit" className="register-btn">
              Tạo tài khoản
            </button>
          </form>

          <div className="register-link">
            Đã có tài khoản?{" "}
            <a href="#" onClick={() => navigate("/login")}>Đăng nhập</a>
          </div>
        </div>
      </div>
    </div>
  );
}

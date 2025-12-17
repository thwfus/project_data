CREATE TABLE IF NOT EXISTS prediction_history (
  id INT AUTO_INCREMENT PRIMARY KEY,

  user_id INT NULL,              -- NULL nếu chưa login
  sex INT,
  age INT,
  bmi FLOAT,
  highbp INT,
  highchol INT,
  smoker INT,
  physactivity INT,

  risk_prob FLOAT NOT NULL,      -- xác suất (0–1 hoặc %)
  risk_label VARCHAR(20),        -- Low / Medium / High

  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  INDEX(user_id),
  CONSTRAINT fk_history_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE SET NULL
);
USE KiThuatDuLieu;

ALTER TABLE prediction_history
  ADD COLUMN cholcheck TINYINT NULL,
  ADD COLUMN stroke TINYINT NULL,
  ADD COLUMN heartdiseaseorattack TINYINT NULL,
  ADD COLUMN fruits TINYINT NULL,
  ADD COLUMN veggies TINYINT NULL,
  ADD COLUMN hvyalcoholconsump TINYINT NULL,
  ADD COLUMN anyhealthcare TINYINT NULL,
  ADD COLUMN nodocbccost TINYINT NULL,
  ADD COLUMN genhlth TINYINT NULL,
  ADD COLUMN menthlth TINYINT NULL,
  ADD COLUMN physhlth TINYINT NULL,
  ADD COLUMN diffwalk TINYINT NULL,
  ADD COLUMN education TINYINT NULL,
  ADD COLUMN income TINYINT NULL;

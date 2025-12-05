import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import Header from "./component/Header";
import Navbar from "./component/Navbar";

import HomePage from "./component/HomePage";
import Info from "./component/Info";

function App() {
  return (
    <BrowserRouter>
      <Header />
      <Navbar />

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/info" element={<Info />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;

import Header from "./Header";
import Navbar from "./Navbar";
import { Outlet } from "react-router-dom";

 function MainLayout() {
  return (
    <>
      <Header />
      {/* <h1>Dự đoán khả năng mắc bệnh tiểu đường</h1> */}
      <Navbar />
      <Outlet />
    </>
  );
}

export default MainLayout;
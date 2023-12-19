import React from "react";

const Navbar = () => {
  return (
    <nav className="fixed z-10 top-0 bg-gray-50 text-gray-800 w-full p-4 grid grid-cols-3 items-center">
      <a href="/" className={`text-center`}>
        Youtube Video Chat 💬
      </a>
    </nav>
  );
};

export default Navbar;

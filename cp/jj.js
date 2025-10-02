const panel = document.querySelector(".result-panel");
if(severity === "ต่ำ") panel.style.background = "#2ecc71";
else if(severity === "ปานกลาง") panel.style.background = "#f1c40f";
else if(severity === "สูง") panel.style.background = "#ff9f1c";
else panel.style.background = "#e74c3c";

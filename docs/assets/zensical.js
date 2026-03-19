const toggle = document.getElementById("theme-toggle");

toggle.addEventListener("click", () => {
  const body = document.body;
  body.dataset.theme = body.dataset.theme === "night" ? "day" : "night";
});

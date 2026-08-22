(function () {
  function slugify(text) {
    return text.trim().toLowerCase()
      .replace(/[^\p{Letter}\p{Number}\s-]/gu, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-");
  }

  document.addEventListener("DOMContentLoaded", function () {
    var content = document.getElementById("post-content");
    var toc = document.getElementById("toc-list");
    if (!content || !toc) return;

    var headings = content.querySelectorAll("h2, h3");
    if (!headings.length) {
      toc.closest("details").hidden = true;
      return;
    }

    var list = document.createElement("ol");
    var used = {};
    headings.forEach(function (heading) {
      var base = heading.id || slugify(heading.textContent) || "section";
      used[base] = (used[base] || 0) + 1;
      heading.id = used[base] > 1 ? base + "-" + used[base] : base;

      var item = document.createElement("li");
      if (heading.tagName === "H3") item.className = "toc-h3";
      var link = document.createElement("a");
      link.href = "#" + heading.id;
      link.textContent = heading.textContent;
      item.appendChild(link);
      list.appendChild(item);
    });
    toc.appendChild(list);
  });
})();


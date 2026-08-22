---
layout: page
title: "Resume"
permalink: /resume/
description: "Resume information."
---

<div class="resume-callout">
  {% if site.resume and site.resume != "" %}
    <p><a href="{{ site.resume | relative_url }}">Download resume (PDF)</a></p>
  {% else %}
    <p>Resume will be available here.</p>
  {% endif %}
</div>

---
layout: page
title: "Resume"
permalink: /en/resume/
description: "Resume information."
lang: en
translation_key: resume
alternate_url: /resume/
---

{% assign t = site.data.i18n[page.lang] %}
<div class="resume-callout">
  {% if site.resume and site.resume != "" %}
    <p><a href="{{ site.resume | relative_url }}">{{ t.common.download_resume }}</a></p>
  {% else %}
    <p>{{ t.common.resume_empty }}</p>
  {% endif %}
</div>

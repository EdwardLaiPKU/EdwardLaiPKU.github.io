---
layout: page
title: "项目"
permalink: /projects/
description: "LLM 系统、AI Agents、Medical AI 与 AI for Science 方向的代表项目。"
lang: zh
translation_key: projects
alternate_url: /en/projects/
---

{% assign current_lang = page.lang | default: 'zh' %}
{% assign t = site.data.i18n[current_lang] %}
{% assign title_key = 'title_' | append: current_lang %}
{% assign description_key = 'description_' | append: current_lang %}
{% assign status_key = 'status_' | append: current_lang %}
{% assign role_key = 'role_' | append: current_lang %}
{% assign contributions_key = 'contributions_' | append: current_lang %}
{% assign published_projects = site.data.projects | where: "published", true %}
{% for project in published_projects %}
<section class="project-item">
  <div class="item-heading">
    <h2>{{ project[title_key] }}</h2>
    <time>{{ project.date }}</time>
  </div>
  {% for paragraph in project[description_key] %}
  <p>{{ paragraph }}</p>
  {% endfor %}
  <p class="meta"><span>{{ t.common.status }}:</span> {{ project[status_key] }} · <span>{{ t.common.role }}:</span> {{ project[role_key] }}</p>

  <h3>{{ t.common.key_contributions }}</h3>
  <ul>
    {% for contribution in project[contributions_key] %}<li>{{ contribution }}</li>{% endfor %}
  </ul>

  <p class="meta"><span>{{ t.common.tech_stack }}:</span> {{ project.tech_stack_zh | join: ' · ' }}</p>
  <p class="text-links">
    {% if project.github and project.github != "" %}<a href="{{ project.github }}">GitHub</a>{% endif %}
    {% if project.article and project.article != "" %}<a href="{{ project.article | relative_url }}">{{ t.common.technical_writing }}</a>{% endif %}
  </p>
</section>
{% endfor %}
{% if published_projects.size == 0 %}
<p class="empty-state">{{ t.common.empty_projects }}</p>
{% endif %}

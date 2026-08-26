---
layout: default
title: "Home"
description: "Zhiwei Lai — LLM, AI Agents, Medical AI and AI for Science."
permalink: /en/
lang: en
translation_key: home
alternate_url: /
---

{% assign t = site.data.i18n[page.lang] %}

## {{ t.sections.about }}

<p class="lede">I am Zhiwei Lai (赖至伟), a Master of Bioinformatics graduate from the University of Melbourne with an undergraduate background in Pharmacy from Peking University.</p>

My current interests lie at the intersection of large language models and biomedical AI, with a focus on AI agents, retrieval-augmented generation, agentic post-training, multimodal systems, and AI for science. I am particularly interested in building reliable LLM systems for medical reasoning, evidence search, and biological sequence modeling.

## {{ t.sections.education }}

<ul class="simple-list">
  <li>
    <span class="date">2023 – 2025</span>
    <strong>University of Melbourne</strong><br>
    <span class="section-note">Master of Bioinformatics · Melbourne, Australia<br>Graduate with Distinction · Dean's List</span>
  </li>
  <li>
    <span class="date">2018 – 2023</span>
    <strong>Peking University</strong><br>
    <span class="section-note">B.S. in Pharmacy · Beijing, China</span>
  </li>
</ul>

## {{ t.sections.projects }}

{% assign published_projects = site.data.projects | where: "published", true %}
{% for project in published_projects limit: 3 %}
  {% include project-item.html project=project %}
{% endfor %}
{% if published_projects.size == 0 %}
<p class="empty-state">{{ t.common.empty_projects }}</p>
{% endif %}

[{{ t.common.view_all_projects }}]({{ '/en/projects/' | relative_url }})

## {{ t.sections.writing }}

{% assign published_posts = site.posts | where: "published", true %}
{% assign recent_posts = published_posts | slice: 0, 4 %}
{% if recent_posts.size > 0 %}
<ul class="post-list">
  {% for post in recent_posts %}
  <li class="post-list-item">
    <div>
      <a class="post-list-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <span class="section-note">· 中文</span>
      <p class="post-list-tags">{{ post.categories | join: ' · ' }}</p>
    </div>
  </li>
  {% endfor %}
</ul>
{% else %}
<p class="empty-state">{{ t.common.empty_writing }}</p>
{% endif %}

[{{ t.common.view_all_writing }}]({{ '/en/writing/' | relative_url }})

## {{ t.sections.contact }}

{% if site.email and site.email != "" %}Email: [{{ site.email }}](mailto:{{ site.email }})  {% endif %}
{% if site.github and site.github != "" %}GitHub: [EdwardLaiPKU]({{ site.github }}){% endif %}

---
layout: default
title: "首页"
description: "Zhiwei Lai 的个人技术主页，关注 LLM、AI Agents、Medical AI 与 AI for Science。"
lang: zh
translation_key: home
alternate_url: /en/
---

{% assign t = site.data.i18n[page.lang] %}

## {{ t.sections.about }}

<div class="about-intro">
  <div class="about-copy">
    <p class="lede">我是赖至伟（Zhiwei Lai），墨尔本大学生物信息学硕士，北京大学药学院理学学士（药学）。</p>
    <p>我的学习和项目经历横跨生命科学与人工智能，目前主要关注大语言模型、AI Agent 与 AI for Science，重点方向包括医疗循证检索、Agent 后训练以及蛋白质与肽序列建模。</p>
    <p>过去一段时间，我主要围绕三个方向做了较完整的项目实践：多智能体医疗循证问答与风险分诊系统、医疗 Deep Search Agent 后训练，以及基于 Protein Language Model 的抗菌肽计算发现系统。相比单纯堆叠模型组件，我更关注系统是否形成清晰的运行时边界、可追踪的实验链路，以及新增模块是否真正带来任务级收益。</p>
  </div>
  {% if site.profile_home_image and site.profile_home_image != "" %}
  <img class="about-photo" src="{{ site.profile_home_image | relative_url }}" alt="赖至伟个人照片" width="210" height="263" decoding="async">
  {% endif %}
</div>

## {{ t.sections.education }}

<ul class="simple-list">
  <li>
    <span class="date">2023 – 2025</span>
    <strong>墨尔本大学</strong><br>
    <span class="section-note">生物信息学硕士 · 墨尔本，澳大利亚<br>Distinction · Dean's List</span>
  </li>
  <li>
    <span class="date">2018 – 2023</span>
    <strong>北京大学药学院</strong><br>
    <span class="section-note">理学学士（药学） · 北京，中国</span>
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

[{{ t.common.view_all_projects }}]({{ '/projects/' | relative_url }})

## {{ t.sections.writing }}

{% assign published_posts = site.posts | where: "published", true %}
{% assign recent_posts = published_posts | slice: 0, 4 %}
{% if recent_posts.size > 0 %}
<ul class="post-list">
  {% for post in recent_posts %}
  <li class="post-list-item">
    <div>
      <a class="post-list-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <p class="post-list-tags">{{ post.categories | join: ' · ' }}</p>
    </div>
  </li>
  {% endfor %}
</ul>
{% else %}
<p class="empty-state">{{ t.common.empty_writing }}</p>
{% endif %}

[{{ t.common.view_all_writing }}]({{ '/writing/' | relative_url }})

## {{ t.sections.contact }}

{% if site.email and site.email != "" %}Email: [{{ site.email }}](mailto:{{ site.email }})  {% endif %}
{% if site.github and site.github != "" %}GitHub: [EdwardLaiPKU]({{ site.github }}){% endif %}

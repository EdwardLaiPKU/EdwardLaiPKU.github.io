---
layout: page
title: "Technical Writing"
permalink: /writing/
description: "Notes on agents, RAG, post-training, multimodal AI, medical AI, and protein language models."
---

{% assign published_posts = site.posts | where: "published", true %}
{% assign posts_by_year = published_posts | group_by_exp: "post", "post.date | date: '%Y'" %}
{% for year in posts_by_year %}
<section class="writing-year">
  <h2>{{ year.name }}</h2>
  <ul class="post-list">
    {% for post in year.items %}
    <li class="post-list-item">
      <time class="post-date" datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: '%b %d' }}</time>
      <div>
        <a class="post-list-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
        {% if post.description %}<p class="post-list-description">{{ post.description }}</p>{% endif %}
        <p class="post-list-tags">{{ post.categories | join: ' · ' }}{% if post.tags.size > 0 %} · {{ post.tags | join: ' · ' }}{% endif %}</p>
      </div>
    </li>
    {% endfor %}
  </ul>
</section>
{% endfor %}
{% if published_posts.size == 0 %}
<p class="empty-state">Technical articles will be published here.</p>
{% endif %}

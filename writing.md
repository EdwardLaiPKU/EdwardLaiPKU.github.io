---
layout: page
title: "Technical Writing"
permalink: /writing/
description: "Notes on agents, RAG, post-training, multimodal AI, medical AI, and protein language models."
---

{% assign published_posts = site.posts | where: "published", true %}
{% if published_posts.size > 0 %}
  <ul class="post-list">
    {% for post in published_posts %}
    <li class="post-list-item">
      <div>
        <a class="post-list-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
        {% if post.description %}<p class="post-list-description">{{ post.description }}</p>{% endif %}
        <p class="post-list-tags">{{ post.categories | join: ' · ' }}{% if post.tags.size > 0 %} · {{ post.tags | join: ' · ' }}{% endif %}</p>
      </div>
    </li>
    {% endfor %}
  </ul>
{% else %}
<p class="empty-state">Technical articles will be published here.</p>
{% endif %}

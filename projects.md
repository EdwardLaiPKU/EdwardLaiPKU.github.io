---
layout: page
title: "Projects"
permalink: /projects/
description: "Selected work in LLM systems, agents, retrieval, medical AI, and AI for Science."
---

{% assign published_projects = site.data.projects | where: "published", true %}
{% for project in published_projects %}
<section class="project-item">
  <div class="item-heading">
    <h2>{{ project.title }}</h2>
    <time>{{ project.date }}</time>
  </div>
  <p>{{ project.description }}</p>
  <p class="meta"><span>Status:</span> {{ project.status }} · <span>Role:</span> {{ project.role }}</p>

  <h3>Key Contributions</h3>
  <ul>
    {% for contribution in project.contributions %}<li>{{ contribution }}</li>{% endfor %}
  </ul>

  <p class="meta"><span>Tech Stack:</span> {{ project.tech_stack | join: ' · ' }}</p>
  <p class="text-links">
    {% if project.github and project.github != "" %}<a href="{{ project.github }}">GitHub</a>{% endif %}
    {% if project.article and project.article != "" %}<a href="{{ project.article | relative_url }}">Technical Writing</a>{% endif %}
  </p>
</section>
{% endfor %}
{% if published_projects.size == 0 %}
<p class="empty-state">Selected projects will be listed here.</p>
{% endif %}

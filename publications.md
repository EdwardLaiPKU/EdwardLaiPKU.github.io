---
layout: page
title: "Publications"
permalink: /publications/
description: "Verified papers, manuscripts, and research artifacts."
---

{% if site.data.publications.size > 0 %}
  {% assign publications_by_year = site.data.publications | group_by: 'year' | sort: 'name' | reverse %}
  {% for year in publications_by_year %}
  <section>
    <h2>{{ year.name }}</h2>
    {% for publication in year.items %}
    <article class="publication">
      <h3>{{ publication.title }}</h3>
      <p>{{ publication.authors }}</p>
      <p class="meta">{{ publication.venue }} · {{ publication.year }}</p>
      <p class="text-links">
        {% if publication.paper and publication.paper != "" %}<a href="{{ publication.paper }}">Paper</a>{% endif %}
        {% if publication.code and publication.code != "" %}<a href="{{ publication.code }}">Code</a>{% endif %}
        {% if publication.project and publication.project != "" %}<a href="{{ publication.project }}">Project</a>{% endif %}
      </p>
    </article>
    {% endfor %}
  </section>
  {% endfor %}
{% else %}
<p class="empty-state">Research outputs and manuscripts will be listed here.</p>
{% endif %}

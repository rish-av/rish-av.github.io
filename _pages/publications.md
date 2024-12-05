---
layout: page
title: Publications
permalink: /publications/
description: A list of my published work
---


<div class="publications-header">
  <h1>Publications</h1>
  <p class="scholar-link">
    <a href="https://scholar.google.com/citations?user=DlT4loUAAAAJ" target="_blank" rel="noopener noreferrer">
      <i class="fas fa-graduation-cap"></i> Google Scholar
    </a>
  </p>
</div>


<style>
.publications-header {
  margin-bottom: 2em;
}

.scholar-link {
  margin-top: 1em;
}

.scholar-link a {
  display: inline-flex;
  align-items: center;
  padding: 0.5em 1em;
  background-color: #f5f5f5;
  border-radius: 4px;
  text-decoration: none;
  color: #333;
  transition: background-color 0.2s;
}

.scholar-link a:hover {
  background-color: #e5e5e5;
}

.scholar-link i {
  margin-right: 0.5em;
}

.publication {
  margin-bottom: 2em;
  padding-bottom: 1em;
  border-bottom: 1px solid #eee;
}
</style>

{% assign years = "2024,2023,2022,2021,2020" | split: "," %}
{% for year in years %}
  {% assign year_publications = site.data.publications | where: "year", year %}
  {% if year_publications.size > 0 %}
    <h2 class="year">{{ year }}</h2>
    {% for pub in year_publications %}
    <div class="publication">
      <div class="publication-title">
        {{ pub.title }}
      </div>
      <div class="publication-authors">
        {{ pub.authors }}
      </div>
      <div class="publication-info">
        <em>{{ pub.venue }}</em>, {{ pub.year }}
        {% if pub.links %}
        <div class="publication-links">
          {% for link in pub.links %}
            <a href="{{ link.url }}" class="publication-link">{{ link.name }}</a>
          {% endfor %}
        </div>
        {% endif %}
      </div>
    </div>
    {% endfor %}
  {% endif %}
{% endfor %}
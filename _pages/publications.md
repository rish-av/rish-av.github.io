---
layout: page
permalink: /publications/
title: Publications
description: 
years: [2024, 2023, 2022, 2020]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">

An up-to-date list is available on <a href="https://scholar.google.com/citations?hl=en&user=DlT4loUAAAAJ">Google Scholar</a>

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f {{ site.scholar.bibliography }} -q @*[year={{y}}]* %}
{% endfor %}

</div>

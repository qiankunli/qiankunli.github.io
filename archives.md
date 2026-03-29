---
title: 归档
permalink: /archives/
layout: single
author_profile: false
---

{% assign posts_by_year = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}

{% for year in posts_by_year %}
## {{ year.name }}

<ul>
  {% for post in year.items %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <small>{{ post.date | date: "%Y/%m/%d" }}</small></li>
  {% endfor %}
</ul>
{% endfor %}

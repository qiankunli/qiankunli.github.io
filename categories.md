---
title: 分类
permalink: /categories/
layout: single
author_profile: false
---

{% assign categories = site.categories | sort %}

{% for category in categories %}
## {{ category[0] }} ({{ category[1].size }})

<ul>
  {% for post in category[1] %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <small>{{ post.date | date: "%Y/%m/%d" }}</small></li>
  {% endfor %}
</ul>
{% endfor %}

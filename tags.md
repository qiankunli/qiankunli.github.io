---
title: 标签
permalink: /tags/
layout: single
author_profile: false
---

{% assign tags = site.tags | sort %}

{% for tag in tags %}
## {{ tag[0] }} ({{ tag[1].size }})

<ul>
  {% for post in tag[1] %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <small>{{ post.date | date: "%Y/%m/%d" }}</small></li>
  {% endfor %}
</ul>
{% endfor %}

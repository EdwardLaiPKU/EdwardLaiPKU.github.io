# Personal AI / LLM Technical Homepage

A minimal, static, Markdown-first personal website for projects, technical writing, publications, and a resume. It is designed for GitHub Pages and uses Jekyll without a JavaScript framework or backend.

## Personalize First

Update `_config.yml` with your verified profile details:

```yaml
name: "Your Name"
name_en: "Your English Name"
title: "Your site title"
email: "you@example.com"
github: "https://github.com/your-account"
linkedin: ""
google_scholar: ""
location: ""
resume: "/assets/resume/resume.pdf"
avatar: "/assets/images/avatar/avatar.jpg"
```

Replace the explicit `TODO` entries in draft posts before publishing them. Add only verified publications to `_data/publications.yml`.

## Bilingual Content

Chinese is the default language at `/`; English static pages live under `/en/`. Shared UI labels are stored in `_data/i18n.yml`, and bilingual project fields remain in the single `_data/projects.yml` file.

Paired static pages declare `lang` and `alternate_url` in Front Matter. Technical articles currently remain Chinese-only: published posts use `lang: zh`, while the English Writing page links to their existing `/writing/.../` URLs.

## Local Development

Ruby and Bundler are required.

```bash
bundle install
bundle exec jekyll serve
```

Open <http://localhost:4000>.

To perform the same production build used for verification:

```bash
JEKYLL_ENV=production bundle exec jekyll build
```

On PowerShell:

```powershell
$env:JEKYLL_ENV = "production"
bundle exec jekyll build
```

## Add a Blog Post

Create `_posts/YYYY-MM-DD-title.md`:

```yaml
---
layout: post
title: "Post title"
date: 2026-08-22 09:00:00 +0800
categories: [Agent, RAG]
tags: [Multi-Agent, Retrieval]
description: "One-sentence description."
draft: true
published: false
lang: zh
math: true
---
```

Write the article in Markdown. The post layout automatically builds a lightweight H2/H3 table of contents. Set `math: true` only when the post uses MathJax. Unpublished files must keep `published: false`; Jekyll will not generate their pages or include them in listings and feeds. After review, change `draft` to `false` and `published` to `true`.

## Add an Image

Store blog images under a topic directory such as:

```text
assets/images/blog/medical-agent/
assets/images/blog/grpo/
assets/images/blog/protein-llm/
```

Reference them from Markdown with a root-relative path:

```markdown
![Architecture diagram](/assets/images/blog/medical-agent/architecture.png)
```

Images are responsive by default. Compress large source images before committing.

## Update Projects and Publications

- Edit `_data/projects.yml` for project entries.
- Edit `_data/publications.yml` for verified papers or manuscripts.
- Set a project's `published` field to `true` only when its displayed content is verified.
- Leave link fields empty until a real destination exists; the templates avoid rendering empty links.

## Update Resume

Replace `assets/resume/resume.pdf`, then set this in `_config.yml`:

```yaml
resume: "/assets/resume/resume.pdf"
```

The Resume navigation item always points to `/resume/`; that page exposes the PDF only after the config value is set.

## Publish on GitHub Pages

1. Create or rename the repository to `<github-username>.github.io`.
2. Update `url`, `repository`, `github`, and other profile fields in `_config.yml`.
3. Push the site to the repository's default branch.
4. In GitHub, open **Settings → Pages** and select **Deploy from a branch**, then choose the default branch and `/ (root)`.
5. Wait for the Pages build, then visit `https://<github-username>.github.io/`.

This repository uses only plugins supported by the `github-pages` gem: `jekyll-feed`, `jekyll-sitemap`, and `jekyll-seo-tag`.

## Repository Structure

```text
_config.yml               Site and profile configuration
_data/                    Navigation, i18n, projects, publications
_includes/                Shared navigation/profile/footer fragments
_layouts/                 Base, page, and post layouts
_posts/                   Markdown technical writing
en/                       English static pages
assets/css/style.scss     Site and responsive styles
assets/js/toc.js          Small H2/H3 table-of-contents generator
assets/images/            Avatar, project, and blog images
assets/resume/            Resume PDF
index.md                  Homepage
projects.md               Projects page
writing.md                Writing archive
publications.md           Publications page
resume.md                 Resume landing page
```

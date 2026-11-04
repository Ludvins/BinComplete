# Build and maintain the documentation website

Read the [hosted documentation](https://ludvins.github.io/BinComplete/).

The website is generated from the Markdown guides in `docs/` with
[Zensical](https://zensical.org/docs/). Edit those guides directly; do not maintain
a second copy of their content in generated HTML. `zensical.toml` defines the
navigation, theme, formulas and diagrams. The README remains the package overview.
The repository's `docs/README.md` is its Markdown guide index; the website uses
`docs/index.md` as its home page and excludes that duplicate index from rendering.

## Preview locally

From the repository root, create a separate documentation environment with
Python 3.12:

```shell
python -m venv .venv-docs
```

Activate it in Windows PowerShell:

```powershell
.venv-docs/Scripts/Activate.ps1
```

Or activate it on Linux or macOS:

```shell
source .venv-docs/bin/activate
```

Then install the pinned documentation dependencies and start the local preview:

```shell
python -m pip install -r requirements-docs.txt
zensical serve
```

Open the local address printed by the command, normally
[http://127.0.0.1:8000](http://127.0.0.1:8000). Changes to guides rebuild the preview.
Press Ctrl+C to stop it. This installs documentation tools only; it does not
install BinComplete, download encoder weights or run any genome analyses.

For a release build with broken-page and anchor checks:

```shell
zensical build --clean --strict
```

The generated site is in `site/`. Keep `site/`, `.cache/` and `.venv-docs/` out of
Git and release archives. Source distributions include the configuration, pinned
documentation requirements and page assets so the site can also be built from
an extracted source archive.

## Editing guides

- Add each new guide to `nav` in `zensical.toml` and to the guide index.
- Use relative Markdown links between guides, including section anchors.
- Link to repository source files with their GitHub URLs. Runtime source files
  are not copied into the published documentation.
- Write formulas with `$...$` or `$$...$$`; MathJax renders them. Diagrams use
  responsive HTML, CSS and decorative SVG connectors. Use `diagram-flow` for a
  sequence and `diagram-grid` for parallel inputs. Keep labels at the normal
  text size and check narrow widths without adding internal scrollbars.
- The home-page workflow uses an HTML ordered list and responsive CSS. Keep its
  source order equal to the process order. The layout follows a reverse C in wide
  containers and stacks vertically in narrow ones without an internal scrollbar.
- Check search, light and dark themes, narrow screens, formulas and diagrams in
  a browser after changing the theme or rendering configuration.

The site uses system fonts and has no configured analytics. Formula rendering
loads the pinned MathJax browser library from jsDelivr. Browser access to that
asset is required for formula rendering. Diagrams use the page HTML and CSS and
need no external diagram runtime. The strict build checks local page and anchor targets, not availability
of external websites or the correctness of scientific explanations.

## Publish with GitHub Pages

The prepared `.github/workflows/docs.yml` validates the site on relevant pull
requests and pushes to `main`. It does not deploy until a maintainer explicitly
enables publication:

1. Review and push the documentation source when ready to publish.
2. In repository **Settings â†’ Pages**, select **GitHub Actions** as the source.
3. In **Settings â†’ Secrets and variables â†’ Actions â†’ Variables**, create the
   repository variable `DOCS_PUBLISH_ENABLED` with the value `true`.
4. Run the **Documentation** workflow on `main`, or push an approved documentation
   change to `main`. Its build must pass before deployment.

After a successful deployment, the site is available at
[https://ludvins.github.io/BinComplete/](https://ludvins.github.io/BinComplete/).
Check the workflow's deployment result after publishing changes.
Pull requests validate documentation without publishing it. Setting
`DOCS_PUBLISH_ENABLED` to `false` stops future deployments; it does not remove
an already published site.

See the [official Zensical publishing guide](https://zensical.org/docs/publish-your-site/)
and [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)
for repository settings and hosting details.

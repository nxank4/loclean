import meta from "../../../pages/_meta.ts";
import concepts_meta from "../../../pages/concepts/_meta.ts";
import getting_started_meta from "../../../pages/getting-started/_meta.ts";
import guides_meta from "../../../pages/guides/_meta.ts";
import reference_meta from "../../../pages/reference/_meta.ts";
export const pageMap = [{
  data: meta
}, {
  name: "concepts",
  route: "/concepts",
  children: [{
    data: concepts_meta
  }, {
    name: "how-it-works",
    route: "/concepts/how-it-works",
    frontMatter: {
      "title": "How It Works",
      "description": "Understanding Loclean's architecture and design principles."
    }
  }]
}, {
  name: "getting-started",
  route: "/getting-started",
  children: [{
    data: getting_started_meta
  }, {
    name: "data-cleaning",
    route: "/getting-started/data-cleaning",
    frontMatter: {
      "title": "Data Cleaning",
      "description": "Clean and normalize data using semantic extraction with clean() function."
    }
  }, {
    name: "installation",
    route: "/getting-started/installation",
    frontMatter: {
      "title": "Installation",
      "description": "Install Loclean for local-first semantic data cleaning."
    }
  }, {
    name: "quick-start",
    route: "/getting-started/quick-start",
    frontMatter: {
      "title": "Quick Start",
      "description": "Get started with Loclean in minutes."
    }
  }]
}, {
  name: "guides",
  route: "/guides",
  children: [{
    data: guides_meta
  }, {
    name: "extraction",
    route: "/guides/extraction",
    frontMatter: {
      "title": "Structured Extraction",
      "description": "Extract structured data from unstructured text with Pydantic schemas."
    }
  }, {
    name: "models",
    route: "/guides/models",
    frontMatter: {
      "title": "Model Management",
      "description": "Download and manage GGUF models for local inference."
    }
  }, {
    name: "performance",
    route: "/guides/performance",
    frontMatter: {
      "title": "Performance Optimization",
      "description": "Tips and best practices for optimizing Loclean performance."
    }
  }, {
    name: "privacy",
    route: "/guides/privacy",
    frontMatter: {
      "title": "Privacy Scrubbing",
      "description": "Scrub sensitive PII data locally using Regex & LLMs."
    }
  }, {
    name: "use-cases",
    route: "/guides/use-cases",
    frontMatter: {
      "title": "Use Cases",
      "description": "Real-world scenarios and examples using Loclean."
    }
  }]
}, {
  name: "index",
  route: "/",
  frontMatter: {
    "title": "Loclean",
    "description": "Local-first Semantic Data Cleaning & Extraction library for Python."
  }
}, {
  name: "reference",
  route: "/reference",
  children: [{
    data: reference_meta
  }, {
    name: "api",
    route: "/reference/api",
    frontMatter: {
      "title": "API Reference",
      "description": "Complete API documentation for Loclean functions."
    }
  }, {
    name: "configuration",
    route: "/reference/configuration",
    frontMatter: {
      "title": "Configuration",
      "description": "Configure Loclean engines, models, and caching."
    }
  }]
}];
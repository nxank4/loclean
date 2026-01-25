---
trigger: glob
glob: "**/README.md"
---

# README Guidelines

## Overview

README files are critical for helping contributors understand the project structure, conventions, and how to work with different parts of the codebase. This document outlines guidelines for maintaining and updating README files across the project.

## README Files in the Project

### 1. Main README (`README.md`)

**Purpose**: User-facing documentation for end users and potential contributors.

**Should Include:**
- Project overview and value proposition
- Installation instructions
- Quick start examples
- Feature highlights
- Available models
- Links to documentation

**Guidelines:**
- Keep it accessible to non-developers
- Use clear, concise language
- Minimize technical jargon (or explain it)
- Include code examples
- Keep it up-to-date with latest features

### 2. Source README (`src/loclean/README.md`)

**Purpose**: Developer documentation for the source code structure.

**Should Include:**
- Directory structure explanation
- Module and file descriptions
- Architecture guidelines
- Development conventions
- Import guidelines
- Resource management rules
- Public API guidelines

**Guidelines:**
- Be detailed and technical
- Explain "why" not just "what"
- Include code examples for conventions
- Link to relevant test files
- Update when architecture changes

### 3. Tests README (`tests/README.md`)

**Purpose**: Testing guidelines and test structure documentation.

**Should Include:**
- Test directory structure
- Test type explanations (unit, scenario, integration)
- Running tests instructions
- Writing test guidelines
- Mocking guidelines
- Coverage requirements
- Pre-commit checklist

**Guidelines:**
- Be comprehensive about test organization
- Include examples of good test patterns
- Explain when to use each test type
- Keep running instructions up-to-date

### 4. Examples README (`examples/README.md`)

**Purpose**: Documentation for example notebooks and scripts.

**Should Include:**
- Available notebooks list
- Getting started instructions
- Requirements
- Running examples
- Contributing guidelines for examples
- Directory structure

**Guidelines:**
- Keep it beginner-friendly
- Include setup instructions
- Explain what each example demonstrates
- Update when adding new examples

## General README Writing Guidelines

### Structure

1. **Title**: Clear, descriptive title
2. **Overview**: Brief description of what this README covers
3. **Main Content**: Organized into logical sections
4. **Examples**: Code examples where relevant
5. **Contributing**: Guidelines for contributors (if applicable)
6. **Links**: Links to related documentation

### Writing Style

- **Be Clear**: Use simple, direct language
- **Be Concise**: Don't repeat information unnecessarily
- **Be Specific**: Give concrete examples, not vague descriptions
- **Be Current**: Keep information up-to-date
- **Be Helpful**: Anticipate reader questions

### Code Examples

- **Keep them simple**: Focus on demonstrating the concept
- **Make them runnable**: Test that examples actually work
- **Add comments**: Explain non-obvious parts
- **Show outputs**: Include expected results when helpful

### Formatting

- **Use headers**: Organize content with proper heading hierarchy
- **Use lists**: Break down complex information into lists
- **Use code blocks**: Format code examples properly
- **Use links**: Link to related documentation
- **Use emphasis**: Bold important points, italicize terms

## When to Update READMEs

### Update Immediately When:

- ✅ Adding new features or modules
- ✅ Changing architecture or structure
- ✅ Updating dependencies or requirements
- ✅ Changing development workflow
- ✅ Adding new test patterns
- ✅ Discovering common issues or questions

### Review Periodically:

- 🔄 Every major release
- 🔄 When onboarding new contributors
- 🔄 When receiving questions that could be answered by README
- 🔄 When project structure evolves

## README Maintenance Checklist

When updating a README:

- [ ] Is the information accurate and current?
- [ ] Are code examples tested and working?
- [ ] Are links valid and pointing to correct locations?
- [ ] Is the structure logical and easy to navigate?
- [ ] Are there any outdated sections that should be removed?
- [ ] Are new features/modules documented?
- [ ] Is the writing clear and accessible?
- [ ] Are there any typos or grammatical errors?

## Contributing to READMEs

### For Contributors

When making changes that affect READMEs:

1. **Update relevant READMEs** as part of your PR
2. **Test code examples** to ensure they work
3. **Follow existing style** and structure
4. **Ask for review** if unsure about documentation approach

### For Reviewers

When reviewing PRs:

1. **Check if READMEs need updates** based on code changes
2. **Verify code examples** are correct and tested
3. **Ensure consistency** with existing documentation style
4. **Suggest improvements** if documentation is unclear

## Common Pitfalls to Avoid

### ❌ Don't:

- Write READMEs that are too technical for their audience
- Include outdated information
- Write code examples that don't work
- Duplicate information across multiple READMEs without linking
- Use jargon without explanation
- Leave sections incomplete or with "TODO"
- Forget to update READMEs when code changes

### ✅ Do:

- Write for your target audience
- Keep information current
- Test all code examples
- Link between related READMEs
- Explain technical terms
- Complete all sections
- Update READMEs as part of code changes

## README Quality Standards

A good README should:

1. **Help readers understand** the purpose and structure
2. **Enable contributors** to work effectively
3. **Reduce questions** by being comprehensive
4. **Stay current** with the codebase
5. **Be discoverable** through clear organization

## Questions?

If you're unsure about README content or structure:

1. **Check existing READMEs** for patterns and style
2. **Ask maintainers** for guidance
3. **Look at similar projects** for inspiration
4. **Start with an outline** and iterate based on feedback

Remember: **Good documentation is as important as good code.**

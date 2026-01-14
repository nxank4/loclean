import { CopyPage } from '@/components/copy-page';
import { getPageImage, source } from '@/lib/source';
import { getMDXComponents } from '@/mdx-components';
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from 'fumadocs-ui/layouts/docs/page';
import { createRelativeLink } from 'fumadocs-ui/mdx';
import type { Metadata } from 'next';
import { notFound } from 'next/navigation';

export default async function Page(props: PageProps<'/docs/[[...slug]]'>) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const MDX = page.data.body;
  
  // Read raw content for CopyPage
  const fs = await import('fs');
  const path = await import('path');
  
  let filePath = (page as any).file?.path;
  if (!filePath && (page as any).info?.path) {
    filePath = (page as any).info.path;
  }
  
  let rawContent = '';
  if (filePath) {
    try {
      const fullPath = path.join(process.cwd(), 'content/docs', filePath);
      rawContent = fs.readFileSync(fullPath, 'utf-8');
    } catch (e) {
      console.error(`Failed to read file at ${filePath}`, e);
    }
  }

  return (
    <DocsPage toc={page.data.toc} full={page.data.full}>
      <div className="flex items-start justify-between">
        <DocsTitle>{page.data.title}</DocsTitle>
        <CopyPage 
          title={page.data.title} 
          description={page.data.description} 
          content={rawContent} 
        />
      </div>
      <DocsDescription>{page.data.description}</DocsDescription>
      <DocsBody>
        <MDX
          components={getMDXComponents({
            // this allows you to link to other pages with relative file paths
            a: createRelativeLink(source, page),
          })}
        />
      </DocsBody>
    </DocsPage>
  );
}

export async function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(props: PageProps<'/docs/[[...slug]]'>): Promise<Metadata> {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  return {
    title: page.data.title,
    description: page.data.description,
    openGraph: {
      images: getPageImage(page).url,
    },
  };
}

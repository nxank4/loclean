import { getLLMText } from '@/lib/get-llm-text';
import { source } from '@/lib/source';
import { notFound } from 'next/navigation';

export const revalidate = false;

export async function GET(
  _req: Request,
  { params }: RouteContext<'/llms.mdx/docs/[[...slug]]'>,
): Promise<Response> {
  const { slug } = await params;
  const page = source.getPage(slug);
  if (!page) notFound();

  return new Response(await getLLMText(page), {
    headers: {
      'Content-Type': 'text/markdown',
    },
  });
}

export function generateStaticParams() {
  const params = source.generateParams();
  return params.filter((param) => {
    // Filter out root
    if (param.slug.length === 0) return false;
    
    // Filter out if it is a parent of another page (file vs directory conflict)
    const currentPath = param.slug.join('/');
    return !params.some((p) => {
      const otherPath = p.slug.join('/');
      return otherPath !== currentPath && otherPath.startsWith(currentPath + '/');
    });
  });
}


import { Star } from 'lucide-react';
import Link from 'next/link';

async function getGitHubStars(repo: string) {
  try {
    const response = await fetch(`https://api.github.com/repos/${repo}`, {
      next: { revalidate: 3600 }, // Cache for 1 hour
    });
    
    if (!response.ok) return null;
    
    const data = await response.json();
    return data.stargazers_count as number;
  } catch (error) {
    return null;
  }
}

export async function GitHubStars() {
  const stars = await getGitHubStars('nxank4/loclean');
  
  return (
    <Link
      href="https://github.com/nxank4/loclean"
      target="_blank"
      className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 text-sm font-medium hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors mb-8"
    >
      <Star className="w-4 h-4 fill-amber-400 text-amber-400" />
      <span>Star on GitHub</span>
      {stars !== null && (
        <>
          <div className="w-px h-4 bg-zinc-200 dark:bg-zinc-800" />
          <span className="tabular-nums">{stars}</span>
        </>
      )}
    </Link>
  );
}

import { GitHubStars } from '@/components/github-stars';
import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center text-center p-8 bg-gradient-to-br from-background to-secondary/20">
      <GitHubStars />
      <h1 className="text-5xl sm:text-7xl font-extrabold tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-violet-600 dark:from-blue-400 dark:to-violet-400">
        Loclean
      </h1>
      <p className="max-w-2xl text-lg sm:text-xl text-muted-foreground mb-8">
        Local-first Data Cleaning & Extraction using LLMs. <br />
        Secure, Private, and Structured.
      </p>
      
      <div className="flex gap-4">
        <Link 
          href="/docs/getting-started" 
          className="px-6 py-3 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 transition-colors shadow-lg shadow-blue-500/20"
        >
          Get Started
        </Link>
        <Link 
          href="https://github.com/nxank4/loclean" 
          target="_blank"
          className="px-6 py-3 rounded-lg border border-zinc-200 dark:border-zinc-800 font-medium hover:bg-zinc-100 dark:hover:bg-zinc-800/50 transition-colors"
        >
          View on GitHub
        </Link>
      </div>

      <div className="mt-20 grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-5xl w-full text-left">
        <div className="p-6 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl hover:-translate-y-1 transition-transform duration-300">
            <h3 className="font-semibold text-lg mb-2">Local Privacy</h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">Everything runs locally. No data leaves your machine. Perfect for PII scrubbing.</p>
        </div>
        <div className="p-6 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl hover:-translate-y-1 transition-transform duration-300">
            <h3 className="font-semibold text-lg mb-2">Structured Output</h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">Get clean, strictly-typed JSON output using Pydantic models. No more parsing errors.</p>
        </div>
        <div className="p-6 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl hover:-translate-y-1 transition-transform duration-300">
            <h3 className="font-semibold text-lg mb-2">DataFrames Friendly</h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">Seamless integration with Pandas and Polars. Automatic batch processing for high throughput.</p>
        </div>
      </div>
    </div>
  );
}

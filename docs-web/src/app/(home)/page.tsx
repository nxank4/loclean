import { ShieldCheck, Table2, Workflow } from 'lucide-react';
import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center text-center p-8 bg-gradient-to-br from-background to-secondary/20 fd-fade-in-up">
      <div className="mt-4 mb-6 flex items-center justify-center gap-4">
        <img
          alt="Loclean Logo"
          src="/logo.svg"
          width={96}
          height={96}
          loading="lazy"
          decoding="async"
          className="logo-light h-24 w-24"
        />
        <img
          alt="Loclean Logo"
          src="/logo-dark.svg"
          width={96}
          height={96}
          loading="lazy"
          decoding="async"
          className="logo-dark h-24 w-24"
        />
        <h1 className="text-5xl sm:text-7xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-violet-600 dark:from-blue-400 dark:to-violet-400 text-left">
          Loclean
        </h1>
      </div>
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
        <div className="p-6 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl hover:-translate-y-1 hover:shadow-xl transition-transform duration-300 ease-out">
          <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-full bg-blue-600/10 text-blue-600 dark:bg-blue-500/15 dark:text-blue-400">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <h3 className="font-semibold text-lg mb-2">Local Privacy</h3>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            Everything runs locally. No data leaves your machine. Perfect for PII scrubbing.
          </p>
        </div>
        <div className="p-6 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl hover:-translate-y-1 hover:shadow-xl transition-transform duration-300 ease-out">
          <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-full bg-violet-600/10 text-violet-600 dark:bg-violet-500/15 dark:text-violet-400">
            <Workflow className="h-5 w-5" />
          </div>
          <h3 className="font-semibold text-lg mb-2">Structured Output</h3>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            Get clean, strictly-typed JSON output using Pydantic models. No more parsing errors.
          </p>
        </div>
        <div className="p-6 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl hover:-translate-y-1 hover:shadow-xl transition-transform duration-300 ease-out">
          <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-full bg-emerald-600/10 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-400">
            <Table2 className="h-5 w-5" />
          </div>
          <h3 className="font-semibold text-lg mb-2">DataFrames Friendly</h3>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            Seamless integration with Pandas and Polars. Automatic batch processing for high throughput.
          </p>
        </div>
      </div>
    </div>
  );
}

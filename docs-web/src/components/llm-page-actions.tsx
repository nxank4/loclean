'use client';

import { Button } from '@/components/ui/button';
import { Check, Copy } from 'lucide-react';
import { useState } from 'react';

interface LLMPageActionsProps {
  markdownUrl: string;
}

export function LLMPageActions({ markdownUrl }: LLMPageActionsProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      const res = await fetch(markdownUrl);
      const text = await res.text();
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // swallow errors; clipboard failures should not crash the docs page
    }
  };

  return (
    <div className="flex items-center gap-2">
      <Button
        onClick={handleCopy}
        variant="secondary"
        size="sm"
        className="h-8 px-3 rounded-full bg-accent/70 text-xs font-medium text-accent-foreground shadow-sm hover:bg-accent hover:shadow-md transition-transform duration-150 ease-out hover:scale-105 active:scale-95"
        aria-label="Copy page as Markdown for AI"
      >
        {copied ? (
          <>
            <Check className="mr-1 h-3.5 w-3.5" />
            <span>Copied</span>
          </>
        ) : (
          <>
            <Copy className="mr-1 h-3.5 w-3.5" />
            <span>AI</span>
          </>
        )}
      </Button>
    </div>
  );
}


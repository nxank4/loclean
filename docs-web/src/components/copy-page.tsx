'use client';

import { Button } from '@/components/ui/button';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Check, Copy, FileText, MessageSquare, Terminal } from 'lucide-react';
import { useState } from 'react';

interface CopyPageProps {
  title: string;
  description?: string;
  content: string;
}

export function CopyPage({ title, description, content }: CopyPageProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (text: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getFormatContent = (format: string) => {
    const header = title ? `# ${title}\n${description ? `> ${description}\n` : ''}\n` : '';
    
    switch (format) {
      case 'markdown':
        return `${header}${content}`;
      case 'plaintext':
        return `${header}${content}`.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').replace(/[*#`]/g, '');
      case 'claude':
        return `Here is the documentation for ${title}:\n\n${header}${content}\n\nPlease explain this section and provide examples.`;
      case 'gpt':
        return `Act as an expert developer. Based on the following documentation for ${title}, answer my questions:\n\n${header}${content}`;
      case 'gemini':
        return `Analyze the following documentation for ${title} and summarize the key points:\n\n${header}${content}`;
      case 'grok':
        return `You are Grok. Read this documentation for ${title} carefully:\n\n${header}${content}\n\nWhat are the most important takeaways?`;
      default:
        return content;
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8 rounded-full bg-background/80 shadow-sm border-border/70 text-muted-foreground hover:text-foreground hover:bg-accent/80 hover:shadow-md transition-transform duration-150 ease-out hover:scale-105 active:scale-95"
        >
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          <span className="sr-only">Copy Page</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuItem onClick={() => handleCopy(getFormatContent('markdown'))}>
          <FileText className="mr-2 h-4 w-4" />
          <span>Copy as Markdown</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleCopy(getFormatContent('plaintext'))}>
          <Terminal className="mr-2 h-4 w-4" />
          <span>Copy as Plaintext</span>
        </DropdownMenuItem>
        <div className="h-px bg-border my-1" />
        <DropdownMenuItem onClick={() => handleCopy(getFormatContent('claude'))}>
          <MessageSquare className="mr-2 h-4 w-4" />
          <span>Copy for Claude</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleCopy(getFormatContent('gpt'))}>
          <MessageSquare className="mr-2 h-4 w-4" />
          <span>Copy for ChatGPT</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleCopy(getFormatContent('gemini'))}>
          <MessageSquare className="mr-2 h-4 w-4" />
          <span>Copy for Gemini</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleCopy(getFormatContent('grok'))}>
          <MessageSquare className="mr-2 h-4 w-4" />
          <span>Copy for Grok</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

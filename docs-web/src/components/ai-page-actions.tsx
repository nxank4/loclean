'use client';

import { Claude, Gemini, Grok, OpenAI } from '@lobehub/icons';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { Check, ChevronDown, Copy, ExternalLink, FileCode, FileText } from 'lucide-react';
import { type ReactNode, useState } from 'react';

type CopyFormat = 'plaintext' | 'markdown' | 'gpt' | 'claude' | 'gemini' | 'grok';

interface PageActionsProps {
  markdownUrl: string;
  githubUrl: string;
  title: string;
}

const FORMAT_CONFIG: Record<CopyFormat, { label: string; icon: ReactNode }> = {
  plaintext: { label: 'Plaintext', icon: <FileText className="h-4 w-4" /> },
  markdown: { label: 'Markdown', icon: <FileCode className="h-4 w-4" /> },
  gpt: {
    label: 'ChatGPT',
    icon: <OpenAI size={16} className="h-4 w-4 text-current" />,
  },
  claude: {
    label: 'Claude',
    icon: <Claude size={16} className="h-4 w-4 text-current" />,
  },
  gemini: {
    label: 'Gemini',
    icon: <Gemini size={16} className="h-4 w-4 text-current" />,
  },
  grok: {
    label: 'Grok',
    icon: <Grok size={16} className="h-4 w-4 text-current" />,
  },
};

function formatContent(content: string, format: CopyFormat, title: string): string {
  switch (format) {
    case 'plaintext':
      return content
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .replace(/[*#`]/g, '');
    case 'markdown':
      return content;
    case 'gpt':
      return `Act as an expert developer. Based on the following documentation for "${title}", answer my questions:\n\n${content}`;
    case 'claude':
      return `Here is the documentation for "${title}":\n\n${content}\n\nPlease explain this section and provide examples.`;
    case 'gemini':
      return `Analyze the following documentation for "${title}" and summarize the key points:\n\n${content}`;
    case 'grok':
      return `You are Grok. Read this documentation for "${title}" carefully:\n\n${content}\n\nWhat are the most important takeaways?`;
    default:
      return content;
  }
}

export function PageActions({ markdownUrl, githubUrl, title }: PageActionsProps) {
  const [copied, setCopied] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState<CopyFormat>('plaintext');

  const handleCopy = async (format: CopyFormat = selectedFormat) => {
    try {
      const res = await fetch(markdownUrl);
      const rawContent = await res.text();
      const formatted = formatContent(rawContent, format, title);
      await navigator.clipboard.writeText(formatted);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard/network failures should not break the page
    }
  };

  const handleSelectFormat = (format: CopyFormat) => {
    setSelectedFormat(format);
    handleCopy(format);
  };

  const currentConfig = FORMAT_CONFIG[selectedFormat];

  return (
    <div className="flex items-center">
      {/* Primary copy button */}
      <button
        onClick={() => handleCopy()}
        className="inline-flex h-8 items-center gap-1.5 rounded-l-md border border-r-0 border-fd-border bg-fd-background px-3 text-xs font-medium text-fd-foreground shadow-sm transition-all duration-150 ease-out hover:bg-fd-accent hover:text-fd-accent-foreground active:scale-[0.98]"
        aria-label={`Copy page as ${currentConfig.label}`}
      >
        {copied ? (
          <>
            <Check className="h-3.5 w-3.5 text-green-500" />
            <span>Copied</span>
          </>
        ) : (
          <>
            <Copy className="h-3.5 w-3.5" />
            <span>Copy</span>
          </>
        )}
      </button>

      {/* Dropdown for format selection */}
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <button
            className="inline-flex h-8 w-8 items-center justify-center rounded-r-md border border-fd-border bg-fd-background text-fd-foreground shadow-sm transition-all duration-150 ease-out hover:bg-fd-accent hover:text-fd-accent-foreground active:scale-[0.98]"
            aria-label="Select copy format"
          >
            <ChevronDown className="h-3.5 w-3.5" />
          </button>
        </DropdownMenu.Trigger>

        <DropdownMenu.Portal>
          <DropdownMenu.Content
            align="end"
            sideOffset={4}
            className="z-50 min-w-[11rem] overflow-hidden rounded-md border border-fd-border bg-fd-popover p-1 text-fd-popover-foreground shadow-md animate-in fade-in-0 zoom-in-95 data-[side=bottom]:slide-in-from-top-2"
          >
            <DropdownMenu.Label className="px-2 py-1.5 text-xs font-semibold text-fd-muted-foreground">
              Copy as
            </DropdownMenu.Label>
            <DropdownMenu.Separator className="mx-1 my-1 h-px bg-fd-border" />

            {(Object.keys(FORMAT_CONFIG) as CopyFormat[]).map((format) => {
              const config = FORMAT_CONFIG[format];
              const isSelected = format === selectedFormat;
              return (
                <DropdownMenu.Item
                  key={format}
                  onSelect={() => handleSelectFormat(format)}
                  className="flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground focus:bg-fd-accent focus:text-fd-accent-foreground"
                >
                  <span className="flex h-4 w-4 items-center justify-center text-fd-muted-foreground">
                    {config.icon}
                  </span>
                  <span className="flex-1">{config.label}</span>
                  {isSelected && <Check className="h-3.5 w-3.5 text-fd-primary" />}
                </DropdownMenu.Item>
              );
            })}

            <DropdownMenu.Separator className="mx-1 my-1 h-px bg-fd-border" />

            <DropdownMenu.Item asChild>
              <a
                href={markdownUrl}
                target="_blank"
                rel="noreferrer"
                className="flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground focus:bg-fd-accent focus:text-fd-accent-foreground"
              >
                <FileCode className="h-4 w-4 text-fd-muted-foreground" />
                <span>View MDX</span>
              </a>
            </DropdownMenu.Item>
            <DropdownMenu.Item asChild>
              <a
                href={githubUrl}
                target="_blank"
                rel="noreferrer"
                className="flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground focus:bg-fd-accent focus:text-fd-accent-foreground"
              >
                <ExternalLink className="h-4 w-4 text-fd-muted-foreground" />
                <span>Edit on GitHub</span>
              </a>
            </DropdownMenu.Item>
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>
    </div>
  );
}

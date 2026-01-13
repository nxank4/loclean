'use client';

import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

export default function Logo() {
  const { theme, systemTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Determine current theme (system theme fallback)
  const currentTheme = mounted ? (theme === 'system' ? systemTheme : theme) : 'light';
  const logoSrc = currentTheme === 'dark' 
    ? '/loclean/loclean-logo-for-dark.svg' 
    : '/loclean/loclean-logo-for-light.svg';

  return (
    <span>
      <img
        src={logoSrc}
        alt="Loclean logo"
        style={{ height: '24px', marginRight: '8px', verticalAlign: 'middle' }}
      />
      <span style={{ fontWeight: 600 }}>Loclean</span>
    </span>
  );
}

import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

// IBM Plex is a non-variable Google font family, so explicit weights are
// required. 400 body / 500 labels / 600 headings covers the type scale.
const uiSans = IBM_Plex_Sans({
  variable: "--font-ui-sans",
  weight: ["400", "500", "600"],
  subsets: ["latin"],
  display: "swap",
});

const dataMono = IBM_Plex_Mono({
  variable: "--font-data-mono",
  weight: ["400", "500", "600"],
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "AgentDS",
  description: "Solo multi-agent AI data scientist platform.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      data-theme="dark"
      className={`${uiSans.variable} ${dataMono.variable} h-full`}
    >
      <body className="min-h-full bg-bg-sunken text-fg antialiased">
        {children}
      </body>
    </html>
  );
}

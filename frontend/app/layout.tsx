import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const uiSans = Inter({
  variable: "--font-ui-sans",
  subsets: ["latin"],
  display: "swap",
});

const dataMono = JetBrains_Mono({
  variable: "--font-data-mono",
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
      <body className="min-h-full bg-canvas-deep text-text antialiased">
        {children}
      </body>
    </html>
  );
}

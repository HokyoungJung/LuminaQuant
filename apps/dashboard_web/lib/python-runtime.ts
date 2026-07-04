import { execFile } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
export const REPO_ROOT = resolve(__dirname, '../../../');

const execFileAsync = promisify(execFile);
const PYTHON_CANDIDATES = [
  resolve(REPO_ROOT, '.venv/bin/python'),
  resolve(REPO_ROOT, '.venv/Scripts/python.exe'),
];

interface PythonRuntime {
  command: string;
  args: string[];
}

function resolvePythonRuntime(): PythonRuntime {
  const override = process.env.LUMINA_DASHBOARD_PYTHON ?? process.env.LUMINA_PYTHON;
  if (override?.trim()) {
    return { command: override.trim(), args: ['-m'] };
  }
  for (const command of PYTHON_CANDIDATES) {
    if (existsSync(command)) {
      return { command, args: ['-m'] };
    }
  }
  return { command: 'uv', args: ['run', 'python', '-m'] };
}

export async function runUvPythonModuleJson<T>(moduleName: string, ...args: string[]): Promise<T> {
  const runtime = resolvePythonRuntime();
  const { stdout } = await execFileAsync(
    runtime.command,
    [...runtime.args, moduleName, ...args],
    {
      cwd: REPO_ROOT,
      encoding: 'utf-8',
      maxBuffer: 32 * 1024 * 1024,
    },
  );
  return JSON.parse(stdout.trim()) as T;
}


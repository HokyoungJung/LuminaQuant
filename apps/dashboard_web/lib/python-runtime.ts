import { execFile } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
export const REPO_ROOT = resolve(__dirname, '../../../');

const execFileAsync = promisify(execFile);

export async function runUvPythonModuleJson<T>(moduleName: string, ...args: string[]): Promise<T> {
  const { stdout } = await execFileAsync('uv', ['run', 'python', '-m', moduleName, ...args], {
    cwd: REPO_ROOT,
    encoding: 'utf-8',
  });
  return JSON.parse(stdout.trim()) as T;
}


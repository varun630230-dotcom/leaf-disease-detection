import { AlertTriangle, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

interface Props {
  reason?: string;
  message?: string;
}

export default function RejectedImageState({ message }: Props) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-10 text-center max-w-lg mx-auto my-12">
      <div className="mx-auto w-12 h-12 bg-rose-50 rounded-full flex items-center justify-center mb-5 text-rose-600 border border-rose-100">
        <AlertTriangle className="w-6 h-6" />
      </div>

      <h2 className="text-xl font-bold text-slate-900 mb-2 uppercase tracking-wide">
        Image Not Supported
      </h2>

      <p className="text-sm text-slate-600 mb-2 font-medium">
        No supported plant leaf detected.
      </p>
      <p className="text-xs text-slate-500 mb-8">
        {message || 'Please upload a clear image of a supported plant leaf.'}
      </p>

      <Link
        to="/"
        className="inline-flex justify-center items-center gap-2 px-5 py-2.5 border border-transparent text-sm font-medium rounded-lg text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500 transition-colors shadow-sm"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Upload Another Image</span>
      </Link>
    </div>
  );
}

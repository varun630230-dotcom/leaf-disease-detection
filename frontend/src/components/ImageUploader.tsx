import React, { useState, useRef } from 'react';
import { UploadCloud, X, ArrowRight, AlertCircle } from 'lucide-react';

interface Props {
  onImageSelected: (file: File) => void;
  isAnalyzing: boolean;
}

export default function ImageUploader({ onImageSelected, isAnalyzing }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const supportedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];

  const validateAndSetFile = (file: File) => {
    setValidationError(null);

    if (!supportedTypes.includes(file.type.toLowerCase())) {
      setValidationError('Unsupported format. Please upload JPG, JPEG, PNG, or WEBP.');
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      setValidationError('File size exceeds 25 MB limit.');
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleRemove = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setValidationError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = () => {
    if (selectedFile && !isAnalyzing) {
      onImageSelected(selectedFile);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="w-full max-w-xl mx-auto">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".jpg,.jpeg,.png,.webp"
        className="hidden"
      />

      {!previewUrl ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-10 sm:p-14 text-center cursor-pointer transition-all duration-150 ${
            dragOver
              ? 'border-emerald-500 bg-emerald-50/50 scale-[0.99]'
              : 'border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50/50 shadow-sm'
          }`}
        >
          <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-emerald-50 flex items-center justify-center text-emerald-600">
            <UploadCloud className="w-7 h-7" />
          </div>

          <h3 className="text-base font-semibold text-slate-900 mb-1">
            Upload a plant leaf image
          </h3>
          <p className="text-sm text-slate-500 mb-5">
            Drag & drop your image here, or
          </p>

          <button
            type="button"
            className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-lg text-slate-700 bg-slate-100 hover:bg-slate-200 border border-slate-200 transition-colors"
          >
            Browse Image
          </button>

          <p className="text-xs text-slate-400 mt-6">
            Supported: JPG, JPEG, PNG, WEBP • Max 25 MB
          </p>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden p-6 space-y-6">
          {/* Image Preview */}
          <div className="relative rounded-lg overflow-hidden bg-slate-950/5 aspect-video sm:aspect-[4/3] flex items-center justify-center border border-slate-100">
            <img
              src={previewUrl}
              alt="Leaf Preview"
              className="max-h-full max-w-full object-contain"
            />
            <button
              type="button"
              onClick={handleRemove}
              disabled={isAnalyzing}
              className="absolute top-3 right-3 w-8 h-8 rounded-full bg-slate-900/70 hover:bg-slate-900 text-white flex items-center justify-center transition-colors shadow-sm disabled:opacity-50"
              title="Remove image"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* File Meta */}
          {selectedFile && (
            <div className="flex items-center justify-between text-xs text-slate-500 px-1">
              <span className="font-medium text-slate-700 truncate max-w-[280px]">
                {selectedFile.name}
              </span>
              <span>{formatFileSize(selectedFile.size)}</span>
            </div>
          )}

          {/* Action Button */}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={isAnalyzing}
            className="w-full py-3.5 px-6 rounded-lg text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 font-medium text-sm transition-colors shadow-sm flex items-center justify-center gap-2 disabled:opacity-60"
          >
            <span>Analyze Leaf</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Client validation error */}
      {validationError && (
        <div className="mt-4 p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-500" />
          <span>{validationError}</span>
        </div>
      )}
    </div>
  );
}
